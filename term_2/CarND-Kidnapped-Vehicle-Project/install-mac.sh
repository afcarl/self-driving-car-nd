brew install openssl libuv cmake
git clone https://github.com/uWebSockets/uWebSockets 
cd uWebSockets
git checkout e94b6e1
patch CMakeLists.txt < ../cmakepatch.txt
mkdir build
export PKG_CONFIG_PATH=/usr/local/opt/openssl/lib/pkgconfig 
cd build
cmake ..
make 
sudo make install
# uwebsocket install into system path,:(
sudo mv /usr/include/uWS /usr/local/include
sudo mv /usr/lib64/libuWS.dylib /usr/local/include
cd ..
cd ..
sudo rm -r uWebSockets
