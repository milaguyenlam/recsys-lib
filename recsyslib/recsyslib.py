from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Optional
from ischedule import schedule, run_loop
import pickle
import os
from datetime import datetime, time, timedelta
from time import sleep

from core.Helpers.helpers import get_nearest_time


class RecommenderModule(ABC):
    @abstractmethod
    def fit(self, training_data):
        pass

    @abstractmethod
    def recommend(self, *args, **kwargs) -> Any:
        pass


class RecommendationScheduler:
    def __init__(
        self,
        recommender_module: RecommenderModule,
        training_data_loader: DataLoader,
        recommendation_input_loader: DataLoader,
        recommendation_result_handler: RecommendationResultHandler,
        scheduler_options: SchedulerOptions,
    ) -> None:
        self.recommender_module = recommender_module
        self.training_data_loader = training_data_loader
        self.recommendation_input_loader = recommendation_input_loader
        self.recommendation_result_handler = recommendation_result_handler
        self.scheduler_options = scheduler_options

    def run(self):
        training_period_in_seconds = self.scheduler_options.training_period_in_hours * 3600
        recommending_period_in_seconds = self.scheduler_options.recommending_period_in_hours * 3600
        schedule(self._train, interval=training_period_in_seconds)
        schedule(self._recommend, interval=recommending_period_in_seconds)

        datetime_of_first_run = self.scheduler_options.datetime_of_first_run
        if datetime_of_first_run:
            # the max() is called in case datetime_of_first_run < datetime.now()
            time_until_first_run = max(
                datetime_of_first_run - datetime.now(), timedelta(seconds=0))
            sleep(time_until_first_run.total_seconds())
        run_loop()

    def _train(self):
        training_data = self.training_data_loader.load()
        self.recommender_module.fit(training_data)

    def _recommend(self):
        input_data = self.recommendation_input_loader.load()
        recommendation_result = self.recommender_module.recommend(input_data)
        self.recommendation_result_handler.handle(recommendation_result)


class SchedulerOptions:
    """
    Args:
        time_of_first_run (str, optional): In format "hh:mm" (ISO format). Defaults to None.
    """

    def __init__(
        self,
        training_period_in_hours: int,
        recommending_period_in_hours: int,
        time_of_first_run: str = None,
    ):
        self.training_period_in_hours = training_period_in_hours
        self.recommending_period_in_hours = recommending_period_in_hours
        self.__time_of_first_run = None
        if time_of_first_run:
            self.__time_of_first_run = time.fromisoformat(time_of_first_run)

    @property
    def datetime_of_first_run(self) -> Optional[datetime]:
        if not self.__time_of_first_run:
            return None
        return get_nearest_time(self.__time_of_first_run)


class DataLoader(ABC):
    @abstractmethod
    def load(self):
        pass


class RecommendationResultHandler(ABC):
    @abstractmethod
    def handle(self, recommendation_result):
        pass
