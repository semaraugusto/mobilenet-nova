source ./circuits/compile.sh;
PTAU_PATH=./powersOfTau28_hez_final_24.ptau;
DEBUG=true;

echo "starting circuit compilation"

cd circuits &&

npm install &&
echo "npm install done" &&

compile head bn128 &&
echo "head compilation done" &&
if [ ! DEBUG ]; then
	compile_phase2 head && # This is very slow and requires a lot of RAM.
	echo "head phase2 done"
fi

compile tail bn128 &&
echo "tail compilation done" &&

if [ ! DEBUG ]; then
	compile_phase2 tail && # This is very slow and requires a lot of RAM
	echo "tail phase2 done!"
fi
#
compile backbone vesta &&
echo -e "backbone done!\n" &&

compile MiMC3D vesta &&
echo -e "Done!\n"
