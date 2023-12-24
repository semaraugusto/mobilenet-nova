source ./circuits/compile.sh;
PTAU_PATH=./powersOfTau28_hez_final_24.ptau;

echo "starting circuit compilation"

cd circuits &&

npm install &&
echo "npm install done" &&

compile head bn128 &&
echo "head compilation done" &&
compile_phase2 head &&
echo "head phase2 done" &&

compile tail bn128 &&
echo "tail compilation done" &&
compile_phase2 tail &&
echo "tail phase2 done!" &&
#
compile backbone vesta &&
echo -e "backbone done!\n" &&

compile MiMC3D vesta &&
echo -e "Done!\n"
