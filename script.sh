run_preprocessing() {
    python Ca_imaging/process_xml.py $1
    suite2p --db $1/db.npy --ops $1/ops.npy
}

run_preprocessing /home/yann/DATA/2020_12_09/TSeries-12092020-1112-003

