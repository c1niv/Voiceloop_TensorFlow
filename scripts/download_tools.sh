#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#THE SOFTWARE.

# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

echo "Downloading all relevant mini programs"
sudo apt-get install zip -y
sudo apt-get install csh -y
sudo apt-get install realpath -y
sudo apt-get install autotools-dev -y
sudo apt-get install automake -y
sudo apt-get install festival espeak -y

echo "Installing phonemizer"
git clone https://github.com/bootphon/phonemizer
cd phonemizer
python setup.py build
sudo -H python setup.py install
python3 setup.py build
sudo -H python3 setup.py install
cd ..


echo "Downloading merlin"
git clone https://github.com/CSTR-Edinburgh/merlin

pushd merlin/tools
./compile_tools.sh
popd

mv merlin/tools/bin tools
rm -rf merlin
