from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class ModelStorage:
    task_id: str
    model_id: str
    root_path: Path

    def get_model_dir(self) -> Path:
        return self.root_path.joinpath(self.task_id).joinpath(self.model_id)

    def save(self, history, model, loss, accuracy):
        model_dir = self.get_model_dir()
        if model_dir.exists():
            print("model dir exists")
            return
        model_dir.mkdir(parents=True)
        # model
        model.save(f'{model_dir}/model.h5')
        with open(f'{model_dir}/model.summary,txt', 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))
        # history
        pd.DataFrame(history.history).to_csv(f'{model_dir}/history.csv')
        acc_file = model_dir.joinpath(f'loss_{"{:.3f}".format(loss)}|accuracy_{"{:.3f}".format(accuracy)}.txt')
        acc_file.touch()

    @staticmethod
    def simple_save(history, model, loss, accuracy, task_id: str = "history_models", path: Path = Path('.')):
        ModelStorage(task_id, ModelStorage.new_model_id(), path).save(history, model, loss, accuracy)

    @staticmethod
    def new_model_id() -> str:
        """
        :return: YY_MM_DD_HH_MM_SS
        """
        import time
        return time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
