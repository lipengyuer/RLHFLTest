
if __name__ == '__main__':

    model_path = r"C:\work\git_codes\xiaohua_server_files\local_modules_for_debug\data\pretrained_models\gpt2-chinese-cluecorpussmall"
    from transformers import BertTokenizer, GPT2LMHeadModel, TextGenerationPipeline

    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    text_generator = TextGenerationPipeline(model, tokenizer)
    res =  text_generator("这是很久之前的事情了", max_length=100, do_sample=True)
    print(res)