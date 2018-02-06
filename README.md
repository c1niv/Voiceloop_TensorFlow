# Voiceloop TensorFlow
Copyright 2017-present, Facebook, Inc.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.


This code is a test in trying to recreate https://github.com/facebookresearch/loop in tensorflow


to run code with our setup:
```bash
bash run_prj.sh
```




## Setup
Requirements: Linux, Python3.5 and Tensorflow 1.4

```bash
git clone https://github.com/facebookresearch/loop.git
cd loop
pip3 install -r scripts/requirements.txt
```

### Data
The data used to train the models in the paper can be downloaded via:
```bash
bash scripts/download_data.sh
```

The script downloads and preprocesses a subset of [VCTK](http://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html). This subset contains speakers with american accent.  

The dataset was preprocessed using [Merlin](http://www.cstr.ed.ac.uk/projects/merlin/) - from each audio clip we extracted vocoder features using the [WORLD](http://ml.cs.yamanashi.ac.jp/world/english/) vocoder. After downloading, the dataset will be located under subfolder ```data``` as follows:

```
loop
├── data
    └── vctk
        ├── norm_info
        │   ├── norm.dat
        ├── numpy_feautres
        │   ├── p294_001.npz
        │   ├── p294_002.npz
        │   └── ...
        └── numpy_features_valid
```

The preprocess pipeline can be executed using the following script by Kyle Kastner: https://gist.github.com/kastnerkyle/cc0ac48d34860c5bb3f9112f4d9a0300.
