# Breast-Cancer-AI-Assistant
> *"There can be life after breast cancer. The prerequisite is early detection."* -  [Ann Jillian](https://en.wikipedia.org/wiki/Ann_Jillian)

Initiated within the frame of MLPC Pipo  Competition, it is a web application based  AI  assistant  useful for  cancerologists  to  determine  whether  patients suffering  or likely to suffer from breast cancer. This is done in 02 steps :
1. Mammography ( considering patient's medical past results )
2. Biopsy ( realizing the tumor's mask from mammogram ).

All backed with Flask.
The full description is presented on [Breast_Cancer.pdf](paper/Breast_Cancer.pdf)
<hr>

## Installation Guide
1. Install python 3.7 (if it is not installed ) :
   - `sudo apt update`
   - `sudo apt install software-properties-common`
   - `sudo add-apt-repository ppa:deadsnakes/ppa`
   - `sudo apt update`
   - `sudo apt install python3.7`
2. Clone the repository : `git clone -b main https://github.com/script-0/Breast-Cancer-AI-Assistant/`
3. Open (Enter in) the directory : `Breast-Cancer-AI-Assistant`
4. Install the dependancies : `pip3 install -r requirements.txt`
5. Run the program:
   * `python3 app.py`
   * The server would be launched. Open the web browser and enter the link `http://127.0.0.1:5000`, now you can use the project

## More to come
- [x] Installation Guide
- [ ] Some Captures
- [x] Deploy
- [ ] For Developpers
