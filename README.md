# Cài đặt môi trường conda 
Tải anaconda và làm theo hướng dẫn: https://www.anaconda.com/download.

# Cài đặt Git LFS 
sudo apt-get install git-lfs

# Git clone Volleyball_Main
git lfs clone https://github.com/TheDuyet1812/Volleyball_Main.git

# Vào Volleyball_Main
cd Volleyball_Main

# Tạo môi trường để cài đặt các thư viện
conda create -n name_package python=3.8

# Chú thích: name_package là tên môi trường, python= 3.8 là phiên bản python 3.8.

# Bật môi trường mới khởi tạo 
conda activate name_package

# Cài đặt pip
conda install pip

# Cài đặt các thư viện cần thiết trong requirement.txt với pip trong môi trường vùa tạo
pip install -r requirements.txt


# Chạy với CPU
python tools/infer.py --weights weights/best_stop_aug_ckpt.pt --source swap_and_count.mp4 --device 'cpu'
# Chạy với GPU
python tools/infer.py --weights weights/best_stop_aug_ckpt.pt --source swap_and_count.mp4
# Chú thích:  weights/best_stop_aug_ckpt.pt là đường dẫn đến model, swap_and_count.mp4 là đường dẫn đến video.
