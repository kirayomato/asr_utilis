from asr_eval import HuggingFaceDataset, Qwen3ASRModel, TextNormalizer, evaluate_asr

# 换数据集：
#   AudioFilesDataset("data/", extensions=[".wav"], max_samples=100)
#   HuggingFaceDatasetLocal("ark_new", "test", max_samples=-1)

# 换模型：from asr_eval import zipformer, paraformer_onnx
#   zipformer("offline_ctc") / zipformer("online_ctc") / paraformer_onnx("paraformer-zh-onnx")

# 多模型对比：一次加载数据，逐个模型评估
#   audio_data, targets = dataset.get_data()
#   evaluate_asr(dataset=dataset, model=model, loaded_data=(audio_data, targets))


if __name__ == "__main__":
    dataset = HuggingFaceDataset("urarik/arknights_voices_zh-processed", split="test")
    normalizer = TextNormalizer()

    models = [
        Qwen3ASRModel("Qwen3-ASR-0.6B"),
    ]

    for model in models:
        try:
            results = evaluate_asr(
                dataset=dataset,
                model=model,
                batch_size=16,
                normalizer=normalizer,
                output_dir="results",
                suffix="new_",
            )
            print(f"CER: {results['cer_percentage']:.2f}%")
        except Exception as e:
            print(f"评估模型 {model.model_id} 失败: {e}")
            continue
