import os
import sys

cmd = 'gz topic -t "/cmd_vel" -m gz.msgs.Twist -p "linear: {x: 0}, angular: {z: 0}"'
if len(sys.argv) == 2:
    if sys.argv[1] == "f": 
        cmd = 'gz topic -t "/cmd_vel" -m gz.msgs.Twist -p "linear: {x: 0.1}, angular: {z: 0}"'
    elif sys.argv[1] == "l": 
        cmd = 'gz topic -t "/cmd_vel" -m gz.msgs.Twist -p "linear: {x: 0}, angular: {z: 0.1}"'
    elif sys.argv[1] == "r": 
        cmd = 'gz topic -t "/cmd_vel" -m gz.msgs.Twist -p "linear: {x: 0}, angular: {z: -0.1}"'
    elif sys.argv[1] == "ff": 
        cmd = 'gz topic -t "/cmd_vel" -m gz.msgs.Twist -p "linear: {x: 0.25}, angular: {z: 0}"'
    elif sys.argv[1] == "fl": 
        cmd = 'gz topic -t "/cmd_vel" -m gz.msgs.Twist -p "linear: {x: 0}, angular: {z: 0.2}"'
    elif sys.argv[1] == "fr": 
        cmd = 'gz topic -t "/cmd_vel" -m gz.msgs.Twist -p "linear: {x: 0}, angular: {z: -0.2}"'

os.system("echo "+cmd)
os.system(cmd)
os.system(cmd)
os.system(cmd)