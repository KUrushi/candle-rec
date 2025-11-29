use candle_core::{Device, Module, Result, Tensor};
use candle_nn::{Embedding, VarBuilder, embedding};

pub struct CollaborativeFilteringModel {
    user_embeddings: Embedding,
    item_embeddings: Embedding,
}

impl CollaborativeFilteringModel {
    pub fn new(
        vb: VarBuilder,
        n_users: usize,
        n_items: usize,
        embedding_dim: usize,
    ) -> Result<Self> {
        let user_embeddings = embedding(n_users, embedding_dim, vb.pp("user_emb"))?;
        let item_embeddings = embedding(n_items, embedding_dim, vb.pp("item_emb"))?;
        Ok(Self {
            user_embeddings,
            item_embeddings,
        })
    }

    pub fn forward(&self, user_ids: &Tensor, item_ids: &Tensor) -> Result<Tensor> {
        // 1. ユーザーIDに対応するEmbeddingを取り出す。(batch, embedding)
        let user_embeddings = self.user_embeddings.forward(user_ids)?;

        // 2. アイテムIDに対応するベクトルを取り出す. (batch, embedding)
        let item_embeddings = self.item_embeddings.forward(item_ids)?;

        let ratings = (user_embeddings * item_embeddings)?;
        ratings.sum(1)
    }
    pub fn device(&self) -> Result<&Device> {
        Ok(self.user_embeddings.embeddings().device())
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::{VarMap, embedding};
    use std::collections::HashMap;

    const N_USERS: usize = 10;
    const N_ITEMS: usize = 5;
    const DIM: usize = 32;
    #[test]
    fn test_build_model() -> Result<()> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);

        let _model = CollaborativeFilteringModel::new(vb, N_USERS, N_ITEMS, DIM)?;

        let v_data = varmap.data().lock().unwrap();
        assert!(
            v_data.contains_key("user_emb.weight"),
            "user_emb.weightが見つかりません"
        );
        assert!(
            v_data.contains_key("item_emb.weight"),
            "item_emb.weightが見つかりません"
        );

        let user_w = v_data.get("user_emb.weight").unwrap();
        assert_eq!(
            user_w.dims(),
            &[N_USERS, DIM],
            "ユーザー埋め込みの形状が正しくありません"
        );

        let item_w = v_data.get("item_emb.weight").unwrap();
        assert_eq!(
            item_w.dims(),
            &[N_ITEMS, DIM],
            "アイテム埋め込みの形状が正しくありません"
        );
        Ok(())
    }

    #[test]
    fn test_forward_pass() -> Result<()> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
        let model = CollaborativeFilteringModel::new(vb, N_USERS, N_ITEMS, DIM)?;
        let user_ids = Tensor::new(&[0u32, 1, 9], &Device::Cpu)?;
        let item_ids = Tensor::new(&[0u32, 4, 2], &Device::Cpu)?;

        let output = model.forward(&user_ids, &item_ids)?;

        assert_eq!(output.dims(), &[3], "出力の形状が違います");
        println!("Output values: {:?}", output.to_vec1::<f32>()?);
        Ok(())
    }
    #[test]
    fn test_forward_with_specific_embeddings() -> Result<()> {
        let device = Device::Cpu;

        let user_weights = Tensor::from_slice(&[1.0f32, 1.0, 2.0, 0.0], (2, 2), &device)?;

        let item_weights = Tensor::from_slice(&[0.5, 2.0, 3.0, 3.0], (2, 2), &device)?;
        let mut tensors = HashMap::new();
        tensors.insert("user_emb.weight".to_string(), user_weights);
        tensors.insert("item_emb.weight".to_string(), item_weights);
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);

        let model = CollaborativeFilteringModel::new(vb, 2, 2, 2)?;
        let user_input = Tensor::new(&[0u32, 1u32], &device)?;
        let item_input = Tensor::new(&[0u32, 1u32], &device)?;

        let result = model.forward(&user_input, &item_input)?;
        let result_scalar = result.to_vec1::<f32>()?;
        assert_eq!(result_scalar, vec![2.5, 6.0]);
        Ok(())
    }
}
