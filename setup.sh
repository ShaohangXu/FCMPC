workspace=$(cd $(dirname $0); pwd)
echo $workspace

cd lib/qpoases

rm -rf build

mkdir build
cd build
cmake ../
make

mv libs/libqpOASES.a $workspace/lib