# Breast-Cancer-AI-Assistant
> *"There can be life after breast cancer. The prerequisite is early detection."* -  [Ann Jillian](https://en.wikipedia.org/wiki/Ann_Jillian)

Initiated within the frame of MLPC Pipo  Competition, it is a web application based  AI  assistant  useful for  cancerologists  to  determine  whether  patients suffering  or likely to suffer from breast cancer. This is done in 02 steps :
1. Mammography ( considering patient's medical past results )
2. Biopsy ( realizing the tumor's mask from mammogram ).

All backed with Flask.
The full description is presented on [Breast_Cancer.pdf](paper/Breast_Cancer.pdf)

## Preview

### Mammography
<div style="width:100%">
   <img src="https://user-images.githubusercontent.com/60468539/156038598-c24236c8-8e57-43b6-8927-16d445dd38c0.png" width="33%" />
   <img src="https://user-images.githubusercontent.com/60468539/156038610-a77ab214-ff42-4eda-8d1d-d2293be8fa66.png" width="33%" />
   <img src="https://user-images.githubusercontent.com/60468539/156038617-b13dbe44-5416-4a62-8aa1-4e9f052ed33c.png" width="33%" />
</div>

### Biopsy
<div style="width:100%">
   <img src="https://user-images.githubusercontent.com/60468539/156037344-555b0d52-d43b-4ce3-9ba2-ffa5433f17d4.png" width="33%" />
   <img src="https://user-images.githubusercontent.com/60468539/156037370-37a16b53-f2d4-471a-8cb9-acfc08113d16.png" width="33%" />
   <img src="https://user-images.githubusercontent.com/60468539/156037374-767a12c3-bad5-49da-a6bd-6167488873cb.png" width="33%" />
</div>

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

## For online deploiement
Use version on `deploy` branch.

## More to come
- [x] Installation Guide
- [x] Some Captures
- [x] Deploy
