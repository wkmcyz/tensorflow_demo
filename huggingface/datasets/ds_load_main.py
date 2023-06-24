from datasets import load_dataset_builder, load_dataset, get_dataset_split_names, get_dataset_config_names

from utils.proxy import set_proxy


def ds_builder():
    ds_builder = load_dataset_builder("rotten_tomatoes")
    print(ds_builder.info.description)
    print(ds_builder.info.features)


def ds_split():
    r = get_dataset_split_names("rotten_tomatoes")
    # ['train', 'validation', 'test']
    print(r)
    # Dataset({
    #     features: ['text', 'label'],
    #     num_rows: 1066
    # })
    dataset = load_dataset("rotten_tomatoes", split="validation")
    print(dataset)


def ds_config():
    configs = get_dataset_config_names("PolyAI/minds14")
    # ['cs-CZ', 'de-DE', 'en-AU', 'en-GB', 'en-US', 'es-ES', 'fr-FR', 'it-IT', 'ko-KR', 'nl-NL', 'pl-PL', 'pt-PT', 'ru-RU', 'zh-CN', 'all']
    print(configs)
    # will raise ValueError: Config name is missing.
    # ds = load_dataset("PolyAI/minds14")
    # print(ds)


def access_ds():
    ds = load_dataset("rotten_tomatoes", split="validation")
    # {'text': 'compassionately explores the seemingly irreconcilable situation between conservative christian parents and their estranged gay and lesbian children .', 'label': 1}
    print(ds[0])
    # compassionately explores the seemingly irreconcilable situation between conservative christian parents and their estranged gay and lesbian children .
    print(ds[0]["text"])
    # 1
    print(ds[0]["label"])
    # {'text': ['compassionately explores the seemingly irreconcilable situation between conservative christian parents and their estranged gay and lesbian children .', 'the soundtrack alone is worth the price of admission .'], 'label': [1, 1]}
    print(ds[0:2])


if __name__ == '__main__':
    set_proxy()
    access_ds()
