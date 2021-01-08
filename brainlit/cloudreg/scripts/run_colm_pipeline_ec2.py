import argparse
import boto3
import paramiko
import os
from functools import partial


# assume boto3 credentials are already configured


def run_command_on_server(command, ssh_key_path, ip_address, username="ubuntu"):

    key = paramiko.RSAKey.from_private_key_file(ssh_key_path)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    # Connect/ssh to an instance
    try:
        # Here 'ip_address' is public IP of EC2
        client.connect(hostname=ip_address, username=username, pkey=key)

        # Execute a command after connecting/ssh to an instance
        stdin, stdout, stderr = client.exec_command(command, get_pty=True)
        for line in iter(stdout.readline, ""):
            print(line, end="")

        # output = stdout.read().decode('utf-8')
        errors = stderr.read().decode("utf-8")

        # close the client connection once the job is done
        client.close()
        return errors

    except Exception as e:
        print(e)


def run_colm_pipeline(
    ssh_key_path,
    instance_id,
    input_s3_path,
    output_s3_path,
    num_channels,
    autofluorescence_channel,
    log_s3_path=None,
    instance_type="r5d.24xlarge",
):
    # get ec2 client
    ec2 = boto3.resource("ec2")

    # stop instance in case it is running
    ec2.meta.client.stop_instances(InstanceIds=[instance_id])
    waiter = ec2.meta.client.get_waiter("instance_stopped")
    waiter.wait(InstanceIds=[instance_id])
    # make sure instance is the right type
    ec2.meta.client.modify_instance_attribute(
        InstanceId=instance_id, Attribute="instanceType", Value=instance_type
    )
    # start instance
    print("starting EC2 instance...")
    ec2.meta.client.start_instances(InstanceIds=[instance_id])
    # wait until instance is started up
    waiter = ec2.meta.client.get_waiter("instance_status_ok")
    waiter.wait(InstanceIds=[instance_id])
    print(f"{instance_type} EC2 instance started")
    # get instance ip address
    instance = ec2.Instance(instance_id)

    # now run command on instance
    # update the code on the instance
    update_command = "mkdir -p ~/ssd1 ~/ssd2; cd CloudReg; git pull;"
    print("updating CloudReg code on EC2 instance...")
    errors_update = run_command_on_server(
        update_command, ssh_key_path, instance.public_ip_address
    )
    # mount ssds command
    command1 = "sudo bash CloudReg/scripts/mount_combined_ssds.sh"
    # colm pipeline command
    command2 = f"time /home/ubuntu/colm_pipeline_env/bin/python CloudReg/scripts/colm_pipeline.py {input_s3_path} {output_s3_path} {num_channels} {autofluorescence_channel} --log_s3_path {log_s3_path}"
    print(command2)
    errors1 = run_command_on_server(command1, ssh_key_path, instance.public_ip_address)
    print(errors1)
    errors2 = run_command_on_server(command2, ssh_key_path, instance.public_ip_address)
    print(errors2)

    # shut down instance
    ec2.meta.client.stop_instances(InstanceIds=[instance_id])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Run COLM pipeline on remote EC2 instance with given input parameters"
    )
    parser.add_argument(
        "-ssh_key_path", help="path to identity file used to ssh into given instance"
    )
    parser.add_argument(
        "-instance_id", help="EC2 Instance ID of instance to run COLM pipeline on."
    )
    parser.add_argument(
        "-input_s3_path",
        help="S3 path to input colm data. Should be of the form s3://<bucket>/<experiment>",
        type=str,
    )
    parser.add_argument(
        "-output_s3_path",
        help="S3 path to store precomputed volume. Precomputed volumes for each channel will be stored under this path. Should be of the form s3://<bucket>/<path_to_precomputed>. The data will be saved at s3://<bucket>/<path_to_precomputed>/CHN0<channel>",
        type=str,
    )
    # parser.add_argument('channel_of_interest', help='Channel of interest in experiment',  type=int)
    parser.add_argument(
        "-num_channels", help="Number of channels in experiment", type=int
    )
    parser.add_argument(
        "-autofluorescence_channel", help="Autofluorescence channel number.", type=int
    )
    parser.add_argument(
        "-log_s3_path",
        help="S3 path at which pipeline intermediates can be stored including bias correctin tile.",
        type=str,
        default="",
    )
    parser.add_argument(
        "--instance_type",
        help="EC2 instance type to run pipeline on. minimum r5d.16xlarge",
        type=str,
        default="r5d.16xlarge",
    )

    args = parser.parse_args()

    run_colm_pipeline(
        args.ssh_key_path,
        args.instance_id,
        args.input_s3_path,
        args.output_s3_path,
        args.num_channels,
        args.autofluorescence_channel,
        args.log_s3_path,
        instance_type=args.instance_type,
    )
