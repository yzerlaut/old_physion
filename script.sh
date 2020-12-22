run_preprocessing() {
    python Ca_imaging/process_xml.py $1
    suite2p --db $1/db.npy --ops $1/ops.npy
}


run_preprocessing /media/yann/Yann/2020_11_10/TSeries-11102020-1605-016
