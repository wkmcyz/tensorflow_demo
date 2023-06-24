from utils.proxy import set_proxy


def ds_tokenize():
    from transformers import AutoTokenizer
    from datasets import load_dataset

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    dataset = load_dataset("rotten_tomatoes", split="train")
    r = tokenizer(dataset[0]["text"])
    # {'input_ids': [101, 1996, 2600, 2003, 16036, 2000, 2022, 1996, 7398, 2301, 1005, 1055, 2047, 1000, 16608, 1000, 1998, 2008, 2002, 1005, 1055, 2183, 2000, 2191, 1037, 17624, 2130, 3618, 2084, 7779, 29058, 8625, 13327, 1010, 3744, 1011, 18856, 19513, 3158, 5477, 4168, 2030, 7112, 16562, 2140, 1012, 102], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
    print(r)

    def tokenization(example):
        return tokenizer(example["text"])

    dataset = dataset.map(tokenization, batched=True)
    # {
    #   'text': [
    #     'the rock is destined to be the 21st century\'s new " conan " and that he\'s going to make a splash even greater than arnold schwarzenegger , jean-claud van damme or steven segal .',
    #     'the gorgeously elaborate continuation of " the lord of the rings " trilogy is so huge that a column of words cannot adequately describe co-writer/director peter jackson\'s expanded vision of j . r . r . tolkien\'s middle-earth .',
    #     'effective but too-tepid biopic'
    #   ],
    #   'label': [
    #     1,
    #     1,
    #     1
    #   ],
    #   'input_ids': [
    #     [      101,      1996,      2600,      2003,      16036,      2000,      2022,      1996,      7398,      2301,      1005,      1055,      2047,      1000,      16608,      1000,      1998,      2008,      2002,      1005,      1055,      2183,      2000,      2191,      1037,      17624,      2130,      3618,      2084,      7779,      29058,      8625,      13327,      1010,      3744,      1011,      18856,      19513,      3158,      5477,      4168,      2030,      7112,      16562,      2140,      1012,      102    ],
    #     [      101,      1996,      9882,      2135,      9603,      13633,      1997,      1000,      1996,      2935,      1997,      1996,      7635,      1000,      11544,      2003,      2061,      4121,      2008,      1037,      5930,      1997,      2616,      3685,      23613,      6235,      2522,      1011,      3213,      1013,      2472,      2848,      4027,      1005,      1055,      4423,      4432,      1997,      1046,      1012,      1054,      1012,      1054,      1012,      23602,      1005,      1055,      2690,      1011,      3011,      1012,      102    ],
    #     [      101,      4621,      2021,      2205,      1011,      8915,      23267,      16012,      24330,      102    ]
    #   ],
    #   'token_type_ids': [
    #     [      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0    ],
    #     [      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0    ],
    #     [      0,      0,      0,      0,      0,      0,      0,      0,      0,      0    ]
    #   ],
    #   'attention_mask': [
    #     [      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1    ],
    #     [      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1    ],
    #     [      1,      1,      1,      1,      1,      1,      1,      1,      1,      1    ]
    #   ]
    # }
    print(dataset[:3])
    # load to tf
    from transformers import DataCollatorWithPadding

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")
    tf_dataset = dataset.to_tf_dataset(
        columns=["input_ids", "token_type_ids", "attention_mask"],
        label_cols=["labels"],
        batch_size=2,
        collate_fn=data_collator,
        shuffle=True
    )
    # <_PrefetchDataset element_spec=({'input_ids': TensorSpec(shape=(None, None), dtype=tf.int64, name=None), 'token_type_ids': TensorSpec(shape=(None, None), dtype=tf.int64, name=None), 'attention_mask': TensorSpec(shape=(None, None), dtype=tf.int64, name=None)}, TensorSpec(shape=(None,), dtype=tf.int64, name=None))>
    print(tf_dataset)


def ds_resample_audio():
    from transformers import AutoFeatureExtractor
    from datasets import load_dataset, Audio

    feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
    # Wav2Vec2FeatureExtractor {
    #   "do_normalize": true,
    #   "feature_extractor_type": "Wav2Vec2FeatureExtractor",
    #   "feature_size": 1,
    #   "padding_side": "right",
    #   "padding_value": 0.0,
    #   "return_attention_mask": false,
    #   "sampling_rate": 16000
    # }
    print(feature_extractor)
    dataset = load_dataset("PolyAI/minds14", "en-US", split="train")
    # {'path': '/Users/wkm/.cache/huggingface/datasets/downloads/extracted/0c9ab2cd08bb3f51fdc74ebb8996d2e599287306c039b7e5057c5d09e3cd5e4e/en-US~JOINT_ACCOUNT/602ba55abb1e6d0fbce92065.wav', 'array': array([ 0.        ,  0.00024414, -0.00024414, ..., -0.00024414,
    #         0.        ,  0.        ]), 'sampling_rate': 8000}
    print(dataset[0]["audio"])

    # resample
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))
    # {'path': '/Users/wkm/.cache/huggingface/datasets/downloads/extracted/0c9ab2cd08bb3f51fdc74ebb8996d2e599287306c039b7e5057c5d09e3cd5e4e/en-US~JOINT_ACCOUNT/602ba55abb1e6d0fbce92065.wav', 'array': array([ 1.70561980e-05,  2.18727539e-04,  2.28099947e-04, ...,
    #         3.43842585e-05, -5.96360042e-06, -1.76846552e-05]), 'sampling_rate': 16000}
    print(dataset[0]["audio"])

    # batched resampling
    def preprocess_function(examples):
        audio_arrays = [x["array"] for x in examples["audio"]]
        inputs = feature_extractor(
            audio_arrays, sampling_rate=feature_extractor.sampling_rate, max_length=16000, truncation=True
        )
        return inputs

    dataset = dataset.map(preprocess_function, batched=True)
    # {
    #   'path': [
    #     '/Users/wkm/.cache/huggingface/datasets/downloads/extracted/0c9ab2cd08bb3f51fdc74ebb8996d2e599287306c039b7e5057c5d09e3cd5e4e/en-US~JOINT_ACCOUNT/602ba55abb1e6d0fbce92065.wav',
    #     '/Users/wkm/.cache/huggingface/datasets/downloads/extracted/0c9ab2cd08bb3f51fdc74ebb8996d2e599287306c039b7e5057c5d09e3cd5e4e/en-US~JOINT_ACCOUNT/602baf24bb1e6d0fbce922a7.wav',
    #     '/Users/wkm/.cache/huggingface/datasets/downloads/extracted/0c9ab2cd08bb3f51fdc74ebb8996d2e599287306c039b7e5057c5d09e3cd5e4e/en-US~JOINT_ACCOUNT/602b9f97963e11ccd901cc32.wav'
    #   ],
    #   'audio': [
    #     {
    #       'path': None,
    #       'array': [
    #         0.00000000e+00,
    #         2.13623047e-04,
    #         2.13623047e-04,
    #         // ...,
    #         3.05175781e-05,
    #         -3.05175781e-05,
    #         -3.05175781e-05
    #       ],
    #       'sampling_rate': 16000
    #     },
    #     {
    #       'path': None,
    #       'array': [
    #         -3.05175781e-05,
    #         1.22070312e-04,
    #         2.44140625e-04,
    #         ...,
    #         1.30310059e-02,
    #         1.11389160e-02,
    #         5.67626953e-03
    #       ],
    #       'sampling_rate': 16000
    #     },
    #     {
    #       'path': None,
    #       'array': [
    #         -3.05175781e-05,
    #         0.00000000e+00,
    #         0.00000000e+00,
    #         ...,
    #         7.93457031e-04,
    #         9.15527344e-04,
    #         5.79833984e-04
    #       ],
    #       'sampling_rate': 16000
    #     }
    #   ],
    #   'transcription': [
    #     'I would like to set up a joint account with my partner',
    #     'Henry County set up a joint account with my wife and where are they at',
    #     "hi I'd like to set up a joint account with my partner I'm not seeing the option to do it on the app so I called in to get some help can I do it over the phone with you and give you the information"
    #   ],
    #   'english_transcription': [
    #     'I would like to set up a joint account with my partner',
    #     'Henry County set up a joint account with my wife and where are they at',
    #     "hi I'd like to set up a joint account with my partner I'm not seeing the option to do it on the app so I called in to get some help can I do it over the phone with you and give you the information"
    #   ],
    #   'intent_class': [
    #     11,
    #     11,
    #     11
    #   ],
    #   'lang_id': [
    #     4,
    #     4,
    #     4
    #   ],
    #   'input_values': [
    #     [
    #       0.014755831100046635
    #       // begin line70
    #       // ... to line 16000+
    #     ],
    #     [
    #       0.014755831100046635
    #       // begin line70
    #       // ... to line 16000+
    #     ],
    #     [
    #       0.014755831100046635
    #       // begin line70
    #       // ... to line 16000+
    #     ]
    #   ]
    # }
    print(dataset[:3])


def ds_data_augmentation():
    from transformers import AutoFeatureExtractor
    from datasets import load_dataset, Image

    feature_extractor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
    dataset = load_dataset("beans", split="train")
    # {'image_file_path': '/Users/wkm/.cache/huggingface/datasets/downloads/extracted/0d02b4bc136a2de68a84df005d6b65ef2c04779595a1c32f1eaba7f58dbf9fe7/train/angular_leaf_spot/angular_leaf_spot_train.0.jpg', 'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=500x500 at 0x16FAC4AF0>, 'labels': 0}
    print(dataset[0])
    from torchvision.transforms import RandomRotation

    rotate = RandomRotation(degrees=(0, 90))
    def transforms(examples):
        examples["pixel_values"] = [rotate(image.convert("RGB")) for image in examples["image"]]
        return examples

    dataset.set_transform(transforms)
    dataset[0]['image'].show()
    dataset[0]['pixel_values'].show()

if __name__ == '__main__':
    set_proxy()
    ds_data_augmentation()
