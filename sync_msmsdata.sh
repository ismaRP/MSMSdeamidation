# Synchornize MSMS data

if [ $1 == 'push' ]; then
  rclone sync MSMSdatasets palaeome:MSMSdatasets -P
elif [ $1 == 'pull' ]; then
  rclone sync palaeome:MSMSdatasets MSMSdatasets -P
fi
