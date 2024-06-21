
# $1 gpuid
# $2 runid

w_cls=1.0
w_cyc=1.0
w_info=0.1
w_div=20.0
div_thresh=0.5
w_tgt=1.0

gen=cnn
interpolation=img

n_tgt=30
max_tgt=29

# run DAI
svroot=saved-digit/${gen}_${interpolation}_${w_cls}_${w_cyc}_${w_info}_${w_div}_${div_thresh}_${w_tgt}_run${2}
python3 DAI.py --gpu $1 --data mnist --gen $gen --interpolation $interpolation --n_tgt ${n_tgt} --tgt_epochs 30 --tgt_epochs_fixg 15 --nbatch 100 --batchsize 128 --lr 1e-4 --w_cls ${w_cls} --w_cyc ${w_cyc} --w_info ${w_info} --w_div ${w_div} --div_thresh ${div_thresh} --w_tgt $w_tgt --ckpt saved-digit/base_run0/best.pkl --svroot ${svroot}

# run CIRL
svroot=saved-digit/${gen}_${interpolation}_${w_cls}_${w_cyc}_${w_info}_${w_div}_${div_thresh}_${w_tgt}_run${2}
python3 CIRL.py --gpu $1 --data mnist --gen $gen --interpolation $interpolation --nbatch 100 --batchsize 128 --lr 1e-4 --div_thresh ${div_thresh} --ckpt saved-digit/base_run0/best.pkl --svroot ${svroot}

# run CACE-ND
svroot=saved-digit/${gen}_${interpolation}_${w_cls}_${w_cyc}_${w_info}_${w_div}_${div_thresh}_${w_tgt}_run${2}
python3 CAE.py --gpu $1 --data mnist --gen $gen --interpolation $interpolation --nbatch 100 --batchsize 128 --lr 1e-4  --div_thresh ${div_thresh} --ckpt saved-digit/base_run0/best.pkl --svroot ${svroot}

# run CSDG
svroot=saved-digit/${gen}_${interpolation}_${w_cls}_${w_cyc}_${w_info}_${w_div}_${div_thresh}_${w_tgt}_run${2}
python3 CSDG.py --gpu $1 --data mnist --gen $gen --interpolation $interpolation --nbatch 100 --batchsize 128 --lr 1e-4 --div_thresh ${div_thresh} --ckpt saved-digit/base_run0/best.pkl --svroot ${svroot}
