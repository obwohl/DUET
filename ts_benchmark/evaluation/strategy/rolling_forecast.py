# -*- coding: utf-8 -*-
import itertools
import time
from typing import List, Optional, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
# NEU: Wir brauchen das 'random' Modul, um zufällige Fenster auszuwählen
import random

from ts_benchmark.evaluation.metrics import regression_metrics
from ts_benchmark.evaluation.strategy.constants import FieldNames
from ts_benchmark.evaluation.strategy.forecasting import ForecastingStrategy
from ts_benchmark.models import ModelFactory
from ts_benchmark.models.model_base import BatchMaker, ModelBase
from ts_benchmark.utils.data_processing import split_before


class RollingForecastEvalBatchMaker:
    # ... (Dieser Teil bleibt unverändert)
    def __init__(
        self,
        series: pd.DataFrame,
        index_list: List[int],
    ):
        self.series = series
        self.index_list = index_list
        self.current_sample_count = 0

    def make_batch_predict(self, batch_size: int, win_size: int) -> dict:
        index_list = self.index_list[
            self.current_sample_count : self.current_sample_count + batch_size
        ]
        series = self.series.values
        windows = sliding_window_view(series, window_shape=(win_size, series.shape[1]))
        predict_batch = windows[np.array(index_list) - win_size]
        predict_batch = np.squeeze(predict_batch, axis=1)
        indexes = self.series.index
        windows_time_stamps = sliding_window_view(indexes, window_shape=win_size)
        time_stamps_batch = windows_time_stamps[np.array(index_list) - win_size]
        self.current_sample_count += len(index_list)
        return {"input": predict_batch, "time_stamps": time_stamps_batch}

    def make_batch_eval(self, horizon: int) -> dict:
        series = self.series.values
        horizons = sliding_window_view(series, window_shape=(horizon, series.shape[1]))
        test_batch = horizons[np.array(self.index_list)]
        return {"target": np.squeeze(test_batch, axis=1)}

    def has_more_batches(self) -> bool:
        return self.current_sample_count < len(self.index_list)


class RollingForecastPredictBatchMaker(BatchMaker):
    # ... (Dieser Teil bleibt unverändert)
    def __init__(self, batch_maker: RollingForecastEvalBatchMaker):
        self._batch_maker = batch_maker

    def make_batch(self, batch_size: int, win_size: int) -> dict:
        return self._batch_maker.make_batch_predict(batch_size, win_size)

    def has_more_batches(self) -> bool:
        return self._batch_maker.has_more_batches()


class RollingForecast(ForecastingStrategy):
    # ... (Dieser Teil bleibt unverändert)
    REQUIRED_CONFIGS = [
        "horizon",
        "tv_ratio",
        "train_ratio_in_tv",
        "stride",
        "num_rollings",
        "save_true_pred",
        "save_plots",
    ]

    @staticmethod
    def _get_index(
        train_length: int, test_length: int, horizon: int, stride: int
    ) -> List[int]:
        data_len = train_length + test_length
        index_list = list(range(train_length, data_len - horizon + 1, stride)) + (
            [data_len - horizon] if (test_length - horizon) % stride != 0 else []
        )
        return index_list

    def _get_split_lens(
        self,
        series: pd.DataFrame,
        meta_info: Optional[pd.Series],
        tv_ratio: float,
    ) -> Tuple[int, int]:
        data_len = int(self._get_meta_info(meta_info, "length", len(series)))
        train_length = int(tv_ratio * data_len)
        test_length = data_len - train_length
        if train_length <= 0 or test_length <= 0:
            raise ValueError(
                "The length of training or testing data is less than or equal to 0"
            )
        return train_length, test_length

    def _execute(
        self,
        series: pd.DataFrame,
        meta_info: Optional[pd.Series],
        model_factory: ModelFactory,
        series_name: str,
    ) -> List:
        model = model_factory()
        if model.batch_forecast.__annotations__.get("not_implemented_batch"):
            return self._eval_sample(series, meta_info, model, series_name)
        else:
            return self._eval_batch(series, meta_info, model, series_name)

    # --- NEU: Dies ist die neue, schlanke Plot-Funktion ---
    def _plot_single_forecast_window(
        self,
        history: pd.DataFrame,
        actuals: pd.DataFrame,
        prediction: np.ndarray,
        quantiles: List[float],
        window_index: int,
        series_name: str,
        save_dir: Path,
    ):
        """
        Plottet und speichert eine einzelne Vorhersage für ein Fenster.
        """
        num_vars = actuals.shape[1]
        # Annahme: Quantile sind [unteres, median, oberes]
        lower_quantile, median_quantile, upper_quantile = quantiles
        median_idx = 1
        lower_idx, upper_idx = 0, 2

        fig, axes = plt.subplots(
            nrows=num_vars, ncols=1, figsize=(15, 5 * num_vars), sharex=True, squeeze=False
        )
        axes = axes.flatten()

        # Zeige einen Teil der Historie für den Kontext (z.B. doppelt so lang wie der Horizont)
        history_to_plot = history.iloc[-1 * len(actuals):]

        for i in range(num_vars):
            ax = axes[i]
            var_name = actuals.columns[i]
            
            # 1. Plotte die Historie
            ax.plot(history_to_plot.index, history_to_plot.iloc[:, i], label="History", color="gray")

            # 2. Plotte die wahren Werte der Zukunft
            ax.plot(actuals.index, actuals.iloc[:, i], label="Actual", color="black", linewidth=1)
            
            # 3. Plotte die Median-Vorhersage
            ax.plot(actuals.index, prediction[:, i, median_idx], label=f"Median Forecast", color="blue")
            
            # 4. Plotte das Konfidenzintervall
            ax.fill_between(
                actuals.index,
                prediction[:, i, lower_idx],
                prediction[:, i, upper_idx],
                color="cyan",
                alpha=0.2,
                label=f"Confidence Interval ({lower_quantile}-{upper_quantile})",
            )
            
            ax.set_title(f"Forecast vs. Actuals for {var_name} (Window {window_index})")
            ax.set_ylabel("Value")
            ax.legend()
            ax.grid(True, which="both", linewidth=0.5)
        
        plt.tight_layout()
        # Stelle sicher, dass das Verzeichnis existiert, bevor gespeichert wird.
        save_dir.mkdir(parents=True, exist_ok=True)
        # Speichere jede Grafik als separate Datei
        fig.savefig(save_dir / f"{series_name}_forecast_window_{window_index}.png")
        plt.close(fig)

    def _eval_sample(
        self,
        series: pd.DataFrame,
        meta_info: Optional[pd.Series],
        model: ModelBase,
        series_name: str,
    ) -> List:
        """
        Die Ausführungspipeline für Stichproben von Prognoseaufgaben.
        """
        horizon = self._get_scalar_config_value("horizon", series_name)
        stride = self._get_scalar_config_value("stride", series_name)
        num_rollings = self._get_scalar_config_value("num_rollings", series_name)
        train_ratio_in_tv = self._get_scalar_config_value(
            "train_ratio_in_tv", series_name
        )
        tv_ratio = self._get_scalar_config_value("tv_ratio", series_name)

        # --- Plotting Setup ---
        # --- NEU: Lade-Pfad für vortrainiertes Modell ---
        try:
            model_checkpoint_path = self._get_scalar_config_value(
                "model_checkpoint_path", series_name
            )
        except Exception:
            model_checkpoint_path = None
        save_plots = self._get_scalar_config_value("save_plots", series_name)
        plots_dir = None
        if save_plots:
            sanitized_series_name = Path(series_name).stem
            run_timestamp = time.strftime("%Y%m%d-%H%M%S")
            plots_dir = Path("results_plots") / f"{sanitized_series_name}_{run_timestamp}"

        train_length, test_length = self._get_split_lens(series, meta_info, tv_ratio)
        train_valid_data, test_data = split_before(series, train_length)

        start_fit_time = time.time()
        if model_checkpoint_path and hasattr(model, "load"):
            print(f"Lade vortrainiertes Modell von: {model_checkpoint_path}")
            model.load(model_checkpoint_path)
        else:
            print("Kein Checkpoint-Pfad angegeben. Starte Training...")
            fit_method = model.forecast_fit if hasattr(model, "forecast_fit") else model.fit
            fit_method(train_valid_data, train_ratio_in_tv=train_ratio_in_tv)
        end_fit_time = time.time()

        eval_scaler = self._get_eval_scaler(train_valid_data, train_ratio_in_tv)

        index_list = self._get_index(train_length, test_length, horizon, stride)
        
        # --- NEU: Wähle 10 zufällige Fenster zum Plotten aus ---
        # Wir erstellen ein Set von Indizes, die wir plotten wollen.
        # min(10, ...) stellt sicher, dass es nicht crasht, wenn es weniger als 10 Fenster gibt.
        num_to_plot = min(10, len(index_list))
        indices_to_plot = set(random.sample(range(len(index_list)), k=num_to_plot))
        print(f"--- Plotting {num_to_plot} random forecast windows. ---")
        
        total_inference_time = 0
        all_test_results = []
        all_rolling_actual = []
        all_rolling_predict = []
        
        # Die Schleife bleibt größtenteils gleich, um Metriken für alle Fenster zu sammeln
        for i, index in itertools.islice(enumerate(index_list), num_rollings):
            train, rest = split_before(series, index)
            test, _ = split_before(rest, horizon)

            start_inference_time = time.time()
            predict = model.forecast(horizon, train) # predict hat shape [horizon, vars, quantiles]
            end_inference_time = time.time()
            total_inference_time += end_inference_time - start_inference_time

            # --- NEU: Bedingter Aufruf der neuen Plot-Funktion ---
            # Prüfe, ob das Plotten aktiviert ist und ob der aktuelle Schleifenindex 'i'
            # in unserem Set der zu plottenden Indizes ist.
            if save_plots and i in indices_to_plot and hasattr(model, "config") and hasattr(model.config, "quantiles"):
                self._plot_single_forecast_window(
                    history=train,
                    actuals=test,
                    prediction=predict,
                    quantiles=model.config.quantiles,
                    window_index=i, # Verwende den Schleifenindex für einen eindeutigen Dateinamen
                    series_name=sanitized_series_name,
                    save_dir=plots_dir,
                )

            # --- Metrik-Berechnung bleibt gleich ---
            predict_for_metrics = predict
            if predict.ndim == 3:
                median_idx = 1
                predict_for_metrics = predict[:, :, median_idx]

            single_series_result = self.evaluator.evaluate(
                test.to_numpy(), predict_for_metrics, eval_scaler, train_valid_data.values
            )
            inference_data = pd.DataFrame(
                predict_for_metrics, columns=test.columns, index=test.index
            )

            all_rolling_actual.append(test)
            all_rolling_predict.append(inference_data)
            all_test_results.append(single_series_result)

    


        # Wir berechnen den Nenner separat und stellen sicher, dass er nie null ist.
        num_evaluated_rollings = min(len(index_list), num_rollings)
        if num_evaluated_rollings > 0:
            average_inference_time = float(total_inference_time) / num_evaluated_rollings
            single_series_results = np.mean(np.stack(all_test_results), axis=0).tolist()
        else:
            average_inference_time = 0.0  # Wenn nichts evaluiert wurde, ist die Inferenzzeit 0.
            single_series_results = [np.nan] * len(self.evaluator.metric_names)

        save_true_pred = self._get_scalar_config_value("save_true_pred", series_name)
        actual_data_encoded = self._encode_data(all_rolling_actual) if save_true_pred else np.nan
        inference_data_encoded = self._encode_data(all_rolling_predict) if save_true_pred else np.nan

        single_series_results += [
            series_name,
            end_fit_time - start_fit_time,
            average_inference_time,
            actual_data_encoded,
            inference_data_encoded,
            "",
        ]
        return single_series_results

    # _eval_batch und der Rest der Klasse bleiben unverändert
    def _eval_batch(
        self,
        series: pd.DataFrame,
        meta_info: Optional[pd.Series],
        model: ModelBase,
        series_name: str,
    ) -> List:
        # ...
        pass

    @staticmethod
    def accepted_metrics() -> List[str]:
        return regression_metrics.__all__

    @property
    def field_names(self) -> List[str]:
        return self.evaluator.metric_names + [
            FieldNames.FILE_NAME,
            FieldNames.FIT_TIME,
            FieldNames.INFERENCE_TIME,
            FieldNames.ACTUAL_DATA,
            FieldNames.INFERENCE_DATA,
            FieldNames.LOG_INFO,
        ]