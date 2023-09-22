This finetuning approach uses AzureML's text-generation-pipeline component.
You can either run this using cli or SDK.
For CLI, first, go through the data generation process, update the yml file then submit the job
``` 
az ml job create abstractive_qna_pipeline.yml
```