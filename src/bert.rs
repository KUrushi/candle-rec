use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use hf_hub::{api::sync::Api, Repo, RepoType};
use candle_transformers::models::bert::{BertModel, Config};

pub struct BertEncoder {
    pub model: BertModel,
    pub tokenizer: tokenizers::Tokenizer,
}

impl BertEncoder {
    pub fn new(device: &Device) -> Result<BertEncoder> {
        let model_id = "sentence-transformers/all-MiniLm-L6-v2".to_string();
        let api = Api::new()?.repo(Repo::new(model_id, RepoType::Model));

        let config_filename = api.get("config.json")?;
        let config_text = std::fs::read_to_string(config_filename)?;
        let config: Config = serde_json::from_str(config_text.as_str())?;

        let tokenizer_filename = api.get("tokenizer.json")?;
        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_filename).expect("tokenizerを読み込めませんでした");
        let weight_filename = api.get("model.safetensors")?;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weight_filename], candle_core::DType::F32, device)?
        };
        let model = BertModel::load(vb, &config)?;

        Ok(Self {
            model,
            tokenizer,
        })
    }

    pub fn encode(&self, sentences: Vec<String>) -> Result<candle_core::Tensor> {
        let n_sentences = sentences.len();
        let tokens = self.tokenizer
            .encode_batch(sentences, true)
            .map_err(|e| anyhow::anyhow!(e))?;

        let token_ids: Vec<u32> = tokens.iter()
            .flat_map(|t| t.get_ids())
            .copied()
            .collect();

        // 系列長を計算 (前トークン数 / 文の数)
        let seq_len = token_ids.len() / n_sentences;
        let input_ids = Tensor::from_vec(token_ids, (n_sentences, seq_len), &self.model.device)?;
        let token_type_ids = Tensor::zeros(input_ids.shape(), candle_core::DType::U32, &self.model.device)?;

        // (batch, seq_len, token_embedding_dim)
        let encoding = self.model.forward(&input_ids, &token_type_ids, None)?;
        let embedding = encoding.mean(1)?;
        Ok(embedding)
    }
}

#[cfg(test)]
mod tests {
    use candle_core::Device;
    use super::*;

    #[test]
    fn test_embedding_shape() -> anyhow::Result<()> {
        let titles = vec!["Toy Story".to_string(), "Star Wars".to_string()];
        let device = Device::Cpu;
        let encoder = BertEncoder::new(&device)?;
        let embeddings = encoder.encode(titles)?;
        let actual = embeddings.shape().dims();

        assert_eq!(actual, &[2, 384]);
        Ok(())
    }
}