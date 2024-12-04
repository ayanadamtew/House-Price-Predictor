# House Price Predictor  

House Price Predictor is a simple Python-based project that predicts house prices using a linear regression model. It features a GUI for user input, enabling easy prediction without terminal input. Additionally, the program displays an interactive graph comparing actual and predicted values while highlighting the user's prediction.  

---

## Features  
- Predict house prices based on user-provided data.  
- User-friendly GUI created with `tkinter` for input.  
- Displays an interactive "Actual vs Predicted Prices" graph with user prediction highlighted.  

---

## Requirements  

Before running the program, ensure you have the following installed:  
- Python 3.x  
- Required Python libraries:  
  - `pandas`  
  - `numpy`  
  - `matplotlib`  
  - `scikit-learn`  

---

## Dataset  

The program uses a CSV file named **Housing.csv**, which contains the following columns:  
- `area` (Square feet)  
- `bedrooms` (Number of bedrooms)  
- `bathrooms` (Number of bathrooms)  
- `stories` (Number of stories)  
- `mainroad` (1 for Yes, 0 for No)  
- `guestroom` (1 for Yes, 0 for No)  
- `basement` (1 for Yes, 0 for No)  
- `hotwaterheating` (1 for Yes, 0 for No)  
- `airconditioning` (1 for Yes, 0 for No)  
- `parking` (Number of parking spaces)  
- `prefarea` (1 for Yes, 0 for No)  
- `furnishingstatus` (`semi-furnished`, `furnished`, or `unfurnished`)  
- `price` (Target variable for house price)  

---

## Steps to Run the Program  

1. **Clone the Repository**  
   ```bash  
   git clone https://github.com/ayanadamtew/house-price-predictor.git  
   cd house-price-predictor  
2. **Install Dependencies**  
   Install the required libraries using `pip`:  
   ```bash  
   pip install pandas numpy matplotlib scikit-learn  
   ```  

3. **Ensure Dataset Availability**  
   - Place the `Housing.csv` dataset in the project directory.  

4. **Run the Program**  
   Execute the program using Python:  
   ```bash  
   python main.py  
   ```  

5. **Use the GUI**  
   - Enter the required house details in the GUI form.  
   - Click the "Predict" button to see the predicted house price.  

6. **View the Graph**  
   - After making a prediction, an interactive graph will appear:  
     - Blue dots: Model's predicted values vs actual values.  
     - Green dot: Your prediction based on input data.  
     - Red line: Perfect prediction line (Actual = Predicted).  

---

## Project Structure  

```
house-price-predictor/  
│  
├── Housing.csv               # Dataset file  
├── main.py                   # Main script to run the program  
└── README.md                 # Project documentation  
```  

---

## Contributing  

Contributions are welcome! If you find a bug or have a feature request, please open an issue or submit a pull request.  

---