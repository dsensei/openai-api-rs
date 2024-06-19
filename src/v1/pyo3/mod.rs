use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct LoraRequest {
    pub lora_id: String,
    pub lora_int_id: i32,
    pub lora_local_path: String,
}
