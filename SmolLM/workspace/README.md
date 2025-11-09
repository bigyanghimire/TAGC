Specify
NCCL_SOCKET_IFNAME
MASTER_ADDR
MACHINE_RANK
in .bashrc

Set up NFS:
https://documentation.ubuntu.com/server/how-to/networking/install-nfs/

Server:
sudo apt install nfs-kernel-server
sudo systemctl start nfs-kernel-server.service
sudo mkdir /srv/train # Point this folder to some big disk
Edit /etc/exports. Add:
/srv/train     *(rw,sync,subtree_check)
sudo exportfs -a
sudo ufw open 111
sudo ufw open 2049


Client:
sudo apt install nfs-common
sudo mkdir /opt/train
#sudo mount 192.168.3.130:/srv/train /opt/train
Add to /etc/fstab:
192.168.1.130:/srv/train /opt/train nfs rsize=8192,wsize=8192,timeo=14,intr
