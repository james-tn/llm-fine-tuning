
az account set --subscription 840b5c5c-3f4a-459a-94fc-6bad2a969f9d
az configure --defaults workspace=ws02ent group=ml

az ml environment create -f environment.yml
