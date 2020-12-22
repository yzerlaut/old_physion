run_preprocessing() {
    python Ca_imaging/process_xml.py $1
    suite2p --db $1/db.npy --ops $1/ops.npy
}


run_preprocessing /home/yann/DATA/2020_11_12/TSeries-11122020-1633-009

