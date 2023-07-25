import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

ssh.connect('nva-hdi-cluster-ssh.azurehdinsight.net', username='sshuser', password='6e!Hp3GPAwQNhF')

#command = 'spark-submit --master yarn ./code.py'
#wget https://nvastockage.blob.core.windows.net/nvabucket/test.py
command = 'wget https://nvastockage.blob.core.windows.net/nvabucket/main.py && spark-submit \
             --master yarn --deploy-mode cluster main.py'
stdin, stdout, stderr = ssh.exec_command(command)

print(stdout.read().decode())

commandelete = 'rm main.py'
stdind, stdoutd, stderrd = ssh.exec_command(commandelete)

ssh.close()