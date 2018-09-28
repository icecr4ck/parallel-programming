echo "SEQUENTIEL"
cd seq
chmod a+x run.sh
make
sleep 1s
echo ""
echo ""
echo "___RUN 1"
./run.sh 1 $1 $2
echo ""
echo ""
echo "___RUN 2"
./run.sh 2 $1 $2
make clean
# Le repertoire ne devrait pas contenir d'executable
echo ""
echo ""
echo "___LS"
ls
cd ..

echo ""
echo ""
echo "PARALLELE"
cd par
chmod a+x run.sh
make
sleep 1s
echo ""
echo ""
echo "___RUN"
./run.sh 1 $1 $2
echo ""
echo ""
echo "___RUN"
./run.sh 2 $1 $2
make clean
# Le repertoire ne devrait pas contenir d'executable
echo ""
echo ""
echo "___LS"
ls
cd ..

#Les executions sequentielle et parallele devraient produire les memes resultats
