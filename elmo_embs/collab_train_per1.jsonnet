local NUM_GRAD_ACC = 2;
local BATCH_SIZE = 128 / NUM_GRAD_ACC;

local BASE_READER = {
        "type": "simple_language_modeling",
        "tokenizer": {
	        "type": "just_spaces"
        },
        "token_indexers": {
          "tokens": {
            "type": "single_id"
          },
          "token_characters": {
            "type": "elmo_characters"
          }
        },
        "max_sequence_length": 170,
        "start_tokens": ["<S>"],
        "end_tokens": ["</S>"],
};

local BASE_LOADER = {
  "max_instances_in_memory": BATCH_SIZE * 100,
  "batch_sampler": {
    "type": "bucket",
    "batch_size": BATCH_SIZE,
  }
};

{
  "dataset_reader": {
    "type": "sharded",
    "base_reader": BASE_READER,
  },
  "train_data_path": "/content/drive/MyDrive/CollabData/elmo/dataset_per1/",
  "vocabulary": {
      "type": "from_files",
      "directory": "/content/drive/MyDrive/CollabData/elmo/vocabulary_per1/",
  },
  "model": {
    "type": "language_model",
    "bidirectional": true,
    "num_samples": 8192,
    
    # Sparse embeddings don't work with DistributedDataParallel.
    "sparse_embeddings": false,
    
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "empty"
        },
        "token_characters": {
            "type": "character_encoding",
            "embedding": {
                "num_embeddings": 262,
                // Same as the Transformer ELMo in Calypso. Matt reports that
                // this matches the original LSTM ELMo as well.
                "embedding_dim": 16
            },
            "encoder": {
                "type": "cnn-highway",
                "activation": "relu",
                "embedding_dim": 16,
                "filters": [
                    [1, 32],
                    [2, 32],
                    [3, 64],
                    [4, 128],
                    [5, 256],
                    [6, 512],
                    [7, 1024]],
                "num_highway": 2,
                "projection_dim": 512,
                "projection_location": "after_highway",
                "do_layer_norm": true
            }
        }
      }
    },
    "dropout": 0.1,
    "contextualizer": {
        "type": "bidirectional_language_model_transformer",
        "input_dim": 512,
        "hidden_dim": 2048,
        "num_layers": 6,
        "dropout": 0.1,
        "input_dropout": 0.1
    }
  },
  "data_loader": BASE_LOADER,
  "trainer": {
    "num_epochs": 8,
    "optimizer": {
      "type": "dense_sparse_adam"
    },
    "learning_rate_scheduler": {
      "type": "noam",
      "model_size": 512,
      "warmup_steps": 6000
    },
    "num_gradient_accumulation_steps": NUM_GRAD_ACC,
  }
}
