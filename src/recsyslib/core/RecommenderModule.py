from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Optional
from ischedule import schedule, run_loop
import pickle
import os
from datetime import datetime
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
        recommendation_result_handler: ResultHandler,
        scheduler_options: SchedulerOptions,
    ) -> None:
        self.recommender_module = recommender_module
        self.training_data_loader = training_data_loader
        self.recommendation_input_loader = recommendation_input_loader
        self.recommendation_result_handler = recommendation_result_handler
        self.scheduler_options = scheduler_options

    def run(self, time_of_first_run: str = None):
        """
        Args:
            time_of_first_run (str, optional): In format "hh:mm". Defaults to None.
        """
        schedule(self.train, interval=self.scheduler_options.training_period_in_hours * 3600)
        schedule(
            self.recommend, interval=self.scheduler_options.recommending_period_in_hours * 3600
        )
        if time_of_first_run:
            exact_time = get_nearest_time(time_of_first_run)
            time_until_first_run = (exact_time - datetime.now()).seconds
            sleep(time_until_first_run)
        run_loop()

    def train(self):
        training_data = self.training_data_loader.load()
        self.recommender_module.fit(training_data)

    def recommend(self):
        input_data = self.recommendation_input_loader.load()
        recommendation_result = self.recommender_module.recommend(input_data)
        self.recommendation_result_handler.handle(recommendation_result)


class DataLoader(ABC):
    @abstractmethod
    def load(self):
        pass


class ResultHandler(ABC):
    @abstractmethod
    def handle(self, recommendation_result):
        pass


class SchedulerOptions:
    def __init__(self, training_period_in_hours: int, recommending_period_in_hours: int):
        self.training_period_in_hours = training_period_in_hours
        self.recommending_period_in_hours = recommending_period_in_hours
