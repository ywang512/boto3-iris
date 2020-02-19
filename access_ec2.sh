#!/bin/bash

ssh -i "accesskey-ec2.pem" ubuntu@ec2-54-173-212-7.compute-1.amazonaws.com "python3" < iris-EDA.py
ssh -i "accesskey-ec2.pem" ubuntu@ec2-54-173-212-7.compute-1.amazonaws.com "python3" < iris-models.py
scp -i "accesskey-ec2.pem" -r ubuntu@ec2-54-173-212-7.compute-1.amazonaws.com:*.png ./
scp -i "accesskey-ec2.pem" -r ubuntu@ec2-54-173-212-7.compute-1.amazonaws.com:*.txt ./
scp -i "accesskey-ec2.pem" -r ubuntu@ec2-54-173-212-7.compute-1.amazonaws.com:*.pth ./
