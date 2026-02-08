Run the following lines in a terminal to create the environment.

python3.10 -m venv my_env  ## Replace my_env with the environment name you prefer ##

Activate your environment. Based on the operation system this line would be slightly different. Please check it online.

Then,

pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements_tethernet.txt
