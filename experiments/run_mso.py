import numpy as np
import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from tsl import logger
from tsl.data import SpatioTemporalDataset, SpatioTemporalDataModule
from tsl.data.preprocessing import StandardScaler
from tsl.experiment import Experiment
from tsl.metrics import torch_metrics
from tsl.nn import models as tsl_models

from lib.datasets import GraphMSO
from lib.nn import models, MultiScalePredictor


def get_model_class(model_str):
    if model_str.startswith('hd_tts'):
        model = models.HDTTSModel
    # Baseline models  ##################################################
    elif model_str == 'rnni':
        model = models.RNNIPredictionModel
    elif model_str == 'grin':
        model = models.GRINPredictionModel
    elif model_str == 'grud':
        model = models.GRUDModel
    # Forecasting models  ###############################################
    elif model_str == 'tts_imp':
        model = models.TimeThenGraphIsoModel
    elif model_str == 'tts_amp':
        model = models.TimeThenGraphAnisoModel
    elif model_str == 'tas_imp':
        model = models.TimeAndGraphIsoModel
    elif model_str == 'tas_amp':
        model = models.TimeAndGraphAnisoModel
    elif model_str == 'rnn':
        model = tsl_models.RNNModel
    elif model_str == 'dcrnn':
        model = tsl_models.DCRNNModel
    elif model_str == 'agcrn':
        model = tsl_models.AGCRNModel
    elif model_str == 'gwnet':
        model = tsl_models.GraphWaveNetModel
    else:
        raise NotImplementedError(f'Model "{model_str}" not available.')
    return model


def run(cfg: DictConfig):
    ########################################
    # Get Dataset                          #
    ########################################
    dataset = GraphMSO(**cfg.dataset.hparams)
    original_mask = dataset.get_mask(dtype=bool, as_dataframe=False)

    # Override mask with injected missing values
    dataset.set_mask(dataset.training_mask)
    mask = dataset.get_mask(dtype=bool, as_dataframe=False)

    # Get connectivity
    adj = dataset.get_connectivity(**cfg.dataset.connectivity)

    covariates = dict()
    if cfg.get('mask_as_exog'):
        covariates.update(u=mask.astype(np.float32))

    # Get data and set missing values to nan
    data = dataset.dataframe()
    masked_data = data.where(mask.reshape(mask.shape[0], -1), np.nan)
    # Fill nan with Last Observation Carried Forward
    data = masked_data.ffill().bfill()

    torch_dataset = SpatioTemporalDataset(target=data,
                                          mask=mask,
                                          covariates=covariates,
                                          connectivity=adj,
                                          horizon=cfg.horizon,
                                          window=cfg.window,
                                          stride=cfg.stride)

    # Add mask to model's inputs as 'input_mask'
    torch_dataset.update_input_map(input_mask=['mask'])

    # Scale input features
    scaler_cfg = cfg.get('scaler')
    if scaler_cfg is not None:
        assert scaler_cfg.method.lower() == 'standard'
        scale_axis = (0,) if scaler_cfg.axis == 'node' else (0, 1)
        transform = dict(target=StandardScaler(axis=scale_axis))
    else:
        transform = None

    dm = SpatioTemporalDataModule(
        dataset=torch_dataset,
        scalers=transform,
        splitter=dataset.get_splitter(**cfg.dataset.splitting),
        mask_scaling=True,
        batch_size=cfg.batch_size,
        workers=cfg.workers
    )
    dm.setup()

    if cfg.model.name == 'grud':
        x_mean = (masked_data.iloc[dm.train_slice].mean().values
                  .reshape(dataset.n_nodes, dataset.n_channels))
        dm.torch_dataset.add_covariate('x_mean', x_mean)

    ########################################
    # Create model                         #
    ########################################

    model_cls = get_model_class(cfg.model.name)

    d_exog = torch_dataset.input_map.u.shape[-1] if 'u' in torch_dataset else 0

    model_kwargs = dict(n_nodes=torch_dataset.n_nodes,
                        input_size=torch_dataset.n_channels,
                        mask_size=1,
                        exog_size=d_exog,
                        output_size=torch_dataset.n_channels,
                        horizon=torch_dataset.horizon)

    model_cls.filter_model_args_(model_kwargs)
    model_kwargs.update(cfg.model.hparams)

    ########################################
    # predictor                            #
    ########################################

    loss_fn = torch_metrics.MaskedMAE()
    if cfg.imputation_loss:
        imputation_loss_fn = torch_metrics.MaskedMAE()
    else:
        imputation_loss_fn = None

    log_metrics = {'mae': torch_metrics.MaskedMAE(),
                   'mse': torch_metrics.MaskedMSE(),
                   'mre': torch_metrics.MaskedMRE()}

    if cfg.get('lr_scheduler') is not None:
        scheduler_class = getattr(torch.optim.lr_scheduler,
                                  cfg.lr_scheduler.name)
        scheduler_kwargs = dict(cfg.lr_scheduler.hparams)
    else:
        scheduler_class = scheduler_kwargs = None

    # setup predictor
    predictor = MultiScalePredictor(
        model_class=model_cls,
        model_kwargs=model_kwargs,
        optim_class=getattr(torch.optim, cfg.optimizer.name),
        optim_kwargs=dict(cfg.optimizer.hparams),
        loss_fn=loss_fn,
        metrics=log_metrics,
        scheduler_class=scheduler_class,
        scheduler_kwargs=scheduler_kwargs,
        scale_target=False if scaler_cfg is None else scaler_cfg.scale_target,
        whiten_prob=cfg.whiten_prob,
        imputation_loss_fn=imputation_loss_fn,
        imputation_loss_weight=cfg.imputation_loss_weight,
        imputation_warm_up=cfg.imputation_warm_up,
    )

    ########################################
    # training                             #
    ########################################

    early_stop_callback = EarlyStopping(
        monitor='val_mae',
        patience=cfg.patience,
        mode='min'
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.run.dir,
        save_top_k=1,
        monitor='val_mae',
        mode='min',
    )

    exp_logger = TensorBoardLogger(save_dir=cfg.run.dir)

    trainer = Trainer(max_epochs=cfg.epochs,
                      limit_train_batches=cfg.train_batches,
                      default_root_dir=cfg.run.dir,
                      logger=exp_logger,
                      accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                      gradient_clip_val=cfg.grad_clip_val,
                      callbacks=[early_stop_callback, checkpoint_callback])

    load_model_path = cfg.get('load_model_path')
    if load_model_path is not None:
        predictor.load_model(load_model_path)
    else:
        trainer.fit(predictor, train_dataloaders=dm.train_dataloader(),
                    val_dataloaders=dm.val_dataloader())
        predictor.load_model(checkpoint_callback.best_model_path)

    predictor.freeze()

    result = checkpoint_callback.best_model_score.item()

    ########################################
    # testing                              #
    ########################################

    trainer.test(predictor, dataloaders=dm.test_dataloader())

    ########################################
    # Test on unmasked data                #
    ########################################

    # Restore original mask
    torch_dataset.set_mask(original_mask)
    # Restore target
    torch_dataset.set_data(dataset.numpy())
    # Add data with imputations as input
    torch_dataset.add_covariate('x', data, 't n f',
                                add_to_input_map=True, preprocess=True)
    # Add again scaler for input
    torch_dataset.add_scaler('x', torch_dataset.scalers['target'])
    # Add mask only to mask input
    torch_dataset.add_covariate('input_mask', mask, 't n f',
                                add_to_input_map=True)

    from torchmetrics import MetricCollection
    predictor.test_metrics = MetricCollection(
        metrics={k: predictor._check_metric(m)
                 for k, m in log_metrics.items()},
        prefix='test_', postfix='_unmasked'
    )
    trainer.test(predictor, dataloaders=dm.test_dataloader())

    return result


if __name__ == '__main__':
    exp = Experiment(run_fn=run, config_path='../config/',
                     config_name='mso')
    res = exp.run()
    logger.info(res)
