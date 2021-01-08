mkdir -p ~/ssd1 ~/ssd2
if [ $(($(lsblk | grep nvme | wc -l) - 2)) -lt $((4)) ]
then
    mkfs.ext4 -E nodiscard -m0 /dev/nvme0n1
    mkfs.ext4 -E nodiscard -m0 /dev/nvme1n1
    mount -o discard /dev/nvme0n1 /home/ubuntu/ssd1
    mount -o discard /dev/nvme1n1 /home/ubuntu/ssd2
    chown ubuntu:ubuntu /home/ubuntu/ssd1
    chown ubuntu:ubuntu /home/ubuntu/ssd2

else 
    pvcreate /dev/nvme0n1 /dev/nvme1n1 /dev/nvme2n1 /dev/nvme3n1
    vgcreate LVMVolGroup /dev/nvme0n1 /dev/nvme1n1 /dev/nvme2n1 /dev/nvme3n1
    lvcreate -l 50%FREE -n vol1 LVMVolGroup
    lvcreate -l 100%FREE -n vol2 LVMVolGroup
    #  create filesystems
    mkfs.ext4 -E nodiscard -m0 /dev/LVMVolGroup/vol1
    mkfs.ext4 -E nodiscard -m0 /dev/LVMVolGroup/vol2
    mount -o discard /dev/LVMVolGroup/vol1 /home/ubuntu/ssd1
    mount -o discard /dev/LVMVolGroup/vol2 /home/ubuntu/ssd2
    chown ubuntu:ubuntu -R /home/ubuntu/ssd1
    chown ubuntu:ubuntu -R /home/ubuntu/ssd2
fi