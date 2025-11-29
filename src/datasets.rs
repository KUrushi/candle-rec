use rand::seq::SliceRandom;
use std::collections::HashMap;
use std::path::PathBuf;
use candle_core::{Tensor, Device};
use serde::{Serialize, Deserialize};
use crate::types::Interaction;

#[derive(Debug, Serialize, Deserialize)]
pub struct IdEncoder {
    map: HashMap<String, usize>,
    reverse_map: HashMap<usize, String>
}

impl IdEncoder {
    /// 文字列のリスト(イテレータ)を受け取ってマッピングを作る
    pub fn new<'a>(ids: impl Iterator<Item=&'a String>) -> Self {
        let mut map = HashMap::new();
        let mut reverse_map = HashMap::new();
        let mut count = 0;
        for id in ids {
            if !map.contains_key(id) {
                map.insert(id.clone(), count);
                reverse_map.insert(count, id.clone());
                count += 1;
            }
        }
        Self { map, reverse_map }
    }

    pub fn encode(&self, id: &str) -> Option<usize> {
        self.map.get(id).copied()
    }

    pub fn decode(&self, idx: usize) -> Option<&str> {
        self.reverse_map.get(&idx).map(|s| s.as_str())
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn save(&self, path: &str) -> anyhow::Result<()> {
        let json_string = serde_json::to_string(self)?;
        let path = PathBuf::from(path);
        std::fs::write(path, json_string)?;
        Ok(())
    }

    pub fn load(path: &str) -> anyhow::Result<Self> {
        let json_string = std::fs::read_to_string(path)?;
        let instance = serde_json::from_str(json_string.as_str())?;
        Ok(instance)
    }
}

#[derive(Clone)]
pub struct TensorDataset {
    pub user_ids: Tensor,
    pub item_ids: Tensor,
    pub ratings: Tensor,
    pub len: usize
}

impl TensorDataset {
    pub fn new(
        interactions: &[Interaction],
        user_encoder: &IdEncoder,
        item_encoder: &IdEncoder,
        device: &Device,
    ) -> anyhow::Result<Self> {
        let mut u_ids = Vec::new();
        let mut i_ids = Vec::new();
        let mut rates = Vec::new();

        for action in interactions {
            if let (Some(u), Some(i)) = (user_encoder.encode(&action.user_id), item_encoder.encode(&action.item_id)) {
                u_ids.push(u as u32);
                i_ids.push(i as u32);
                rates.push(action.rating);
            }
        }

        let len = u_ids.len();
        Ok(Self {
            user_ids: Tensor::new(u_ids, device)?,
            item_ids: Tensor::new(i_ids, device)?,
            ratings: Tensor::new(rates, device)?,
            len
        })
    }
}

pub struct DataLoader {
    dataset: TensorDataset,
    batch_size: usize,
    current_index: usize,
    shuffled_indices: Vec<usize>
}
impl DataLoader {
    pub fn new(dataset: TensorDataset, batch_size: usize) -> Self {
        let mut rng = rand::rng();
        let mut shuffled_indices = (0..dataset.len).collect::<Vec<usize>>();
        shuffled_indices.shuffle(&mut rng);

        DataLoader {
            dataset,
            batch_size,
            current_index: 0usize,
            shuffled_indices
        }
    }
}
impl Iterator for DataLoader {
    /// user_ids, item_ids, ratings
    type Item = (Tensor, Tensor, Tensor);
    fn next(&mut self) -> Option<Self::Item> {
        // 1. もうデータがない場合にはNoneを返す
        if self.current_index >= self.shuffled_indices.len() {
            return None;
        }

        // 2. バッチの終わりのインデックスを計算 (データの終わりを超えないように)
        let remaining = self.shuffled_indices.len() - self.current_index;
        let size = self.batch_size.min(remaining);
        let end = self.current_index + size;

        // 3. 今回使うインデックスのリストを取得(Vec<usize>)
        let batch_indices_vec = &self.shuffled_indices[self.current_index..end];

        // 4. indices を Tensorに変換 (deviceはdatasetのものを使うと良い)
        let batch_indices_u32: Vec<u32> = batch_indices_vec.iter().map(|&x| x as u32).collect();

        let device = self.dataset.user_ids.device();
        let indices_tensor = Tensor::new(&*batch_indices_u32, device).expect("ランダムなインデックスの取得に失敗しました");

        // 5. index_selectでデータを抽出
        let users = self.dataset.user_ids.index_select(&indices_tensor,0).expect("user_idsの取得に失敗");
        let items = self.dataset.item_ids.index_select(&indices_tensor, 0).expect("item_idsの取得に失敗");
        let ratings = self.dataset.ratings.index_select(&indices_tensor, 0).expect("ratingsの取得に失敗");

        self.current_index += size;

        Some((users, items, ratings))
    }
}
pub fn split_data(mut data: Vec<Interaction>, split_rate: f32) -> (Vec<Interaction>, Vec<Interaction>) {
    data.sort_by_key(|x| x.timestamp);
    let split_index = ( data.len() as f32 * split_rate) as usize;
    let test_data = data.split_off(split_index);
    (data, test_data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn create_dummy_dataset(size: usize) -> TensorDataset {
        let device = Device::Cpu;
        let users: Vec<u32> = (0..size as u32).collect();
        // 整合性チェックのため、Item ID は User ID の 10倍にしておく
        let items: Vec<u32> = (0..size as u32).map(|x| x * 10).collect();
        let ratings: Vec<f32> = vec![1.0; size];

        TensorDataset {
            // Tensor::new(&users, ...) ではなく from_vec を使う
            // 引数: (データの実体, 形状(1次元なのでsize), デバイス)
            user_ids: Tensor::from_vec(users, size, &device).unwrap(),
            item_ids: Tensor::from_vec(items, size, &device).unwrap(),
            ratings: Tensor::from_vec(ratings, size, &device).unwrap(),
            len: size,
        }
    }

    #[test]
    fn test_dataloader_batching() {
        let data_len = 10;
        let batch_size = 3;
        let dataset = create_dummy_dataset(data_len);

        let loader = DataLoader::new(dataset, batch_size);

        let mut total_count = 0;
        let mut batch_count = 0;

        for (u,i,r) in loader {
            let current_batch_size = u.dims()[0];

            assert_eq!(u.dims()[0], i.dims()[0]);
            assert_eq!(u.dims()[0], r.dims()[0]);

            if batch_count < 3 {
                assert_eq!(current_batch_size, batch_size)
            } else {
                assert_eq!(current_batch_size, 1)

            }
            total_count += current_batch_size;
            batch_count += 1;
        }
        assert_eq!(total_count, data_len);
        assert_eq!(batch_count, 4);
    }

    #[test]
    fn test_shuffle_integrity() {
        let data_len = 50;
        let batch_size = 10;
        let dataset = create_dummy_dataset(data_len);
        let loader = DataLoader::new(dataset, batch_size);

        for (batch_users, batch_items, _) in loader {
            let u_vec = batch_users.to_vec1::<u32>().unwrap();
            let i_vec = batch_items.to_vec1::<u32>().unwrap();

            for (u, i)in u_vec.iter().zip(i_vec.iter()) {
                assert_eq!(*i, u*10, "シャッフルでデータの対応関係が壊れています!");
            }
        }
    }


}