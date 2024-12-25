#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iomanip> // For formatting
#include <chrono> // For measuring execution time
#include <immintrin.h> // Include AVX intrinsic functions

#define inv_sqrt_2xPI 0.39894228040143270286

struct OptionData {
    double spotPrice;
    double strikePrice;
    double riskFreeRate;
    double volatility;
    double timeToMaturity;
    int optionType;
};

float CNDF (float InputX) 
{
    int sign;

    float OutputX;
    float xInput;
    float xNPrimeofX;
    float expValues;
    float xK2;
    float xK2_2, xK2_3;
    float xK2_4, xK2_5;
    float xLocal, xLocal_1;
    float xLocal_2, xLocal_3;

    // Check for negative value of InputX
    if (InputX < 0.0) {
        InputX = -InputX;
        sign = 1;
    } else 
        sign = 0;

    xInput = InputX;
 
    // Compute NPrimeX term common to both four & six decimal accuracy calcs
    expValues = exp(-0.5f * InputX * InputX);
    xNPrimeofX = expValues;
    xNPrimeofX = xNPrimeofX * inv_sqrt_2xPI;

    xK2 = 0.2316419 * xInput;
    xK2 = 1.0 + xK2;
    xK2 = 1.0 / xK2;
    xK2_2 = xK2 * xK2;
    xK2_3 = xK2_2 * xK2;
    xK2_4 = xK2_3 * xK2;
    xK2_5 = xK2_4 * xK2;
    
    xLocal_1 = xK2 * 0.319381530;
    xLocal_2 = xK2_2 * (-0.356563782);
    xLocal_3 = xK2_3 * 1.781477937;
    xLocal_2 = xLocal_2 + xLocal_3;
    xLocal_3 = xK2_4 * (-1.821255978);
    xLocal_2 = xLocal_2 + xLocal_3;
    xLocal_3 = xK2_5 * 1.330274429;
    xLocal_2 = xLocal_2 + xLocal_3;

    xLocal_1 = xLocal_2 + xLocal_1;
    xLocal   = xLocal_1 * xNPrimeofX;
    xLocal   = 1.0 - xLocal;

    OutputX  = xLocal;
    
    if (sign) {
        OutputX = 1.0 - OutputX;
    }
    
    return OutputX;
} 


float blackScholes(float sptprice, float strike, float rate, float volatility,
                   float otime, int otype, float timet)
{
    float OptionPrice;

    // local private working variables for the calculation
    float xStockPrice;
    float xStrikePrice;
    float xRiskFreeRate;
    float xVolatility;
    float xTime;
    float xSqrtTime;

    float logValues;
    float xLogTerm;
    float xD1; 
    float xD2;
    float xPowerTerm;
    float xDen;
    float d1;
    float d2;
    float FutureValueX;
    float NofXd1;
    float NofXd2;
    float NegNofXd1;
    float NegNofXd2;    
    
    xStockPrice = sptprice;
    xStrikePrice = strike;
    xRiskFreeRate = rate;
    xVolatility = volatility;

    xTime = otime;
    xSqrtTime = sqrt(xTime);

    logValues = log(sptprice / strike);
        
    xLogTerm = logValues;
        
    
    xPowerTerm = xVolatility * xVolatility;
    xPowerTerm = xPowerTerm * 0.5;
        
    xD1 = xRiskFreeRate + xPowerTerm;
    xD1 = xD1 * xTime;
    xD1 = xD1 + xLogTerm;

    xDen = xVolatility * xSqrtTime;
    xD1 = xD1 / xDen;
    xD2 = xD1 - xDen;

    d1 = xD1;
    d2 = xD2;
    
    NofXd1 = CNDF(d1);
    NofXd2 = CNDF(d2);

    FutureValueX = strike * (exp(-(rate)*(otime)));        
    if (otype == 0) {            
        OptionPrice = (sptprice * NofXd1) - (FutureValueX * NofXd2);
    } else { 
        NegNofXd1 = (1.0 - NofXd1);
        NegNofXd2 = (1.0 - NofXd2);
        OptionPrice = (FutureValueX * NegNofXd2) - (sptprice * NegNofXd1);
    }
    
    return OptionPrice;
}

// Function to read CSV file and parse the data into a vector of OptionData
std::vector<OptionData> readCSV(const std::string& filename) {
    std::vector<OptionData> options;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return options;
    }

    std::string line;
    bool isHeader = true; // Skip the header line

    while (std::getline(file, line)) {
        if (isHeader) {
            isHeader = false;
            continue; // Skip header
        }

        std::stringstream ss(line);
        std::string token;
        OptionData data;

        // Read each value separated by commas
        std::getline(ss, token, ',');
        data.spotPrice = std::stod(token); // Convert to double

        std::getline(ss, token, ',');
        data.strikePrice = std::stod(token);

        std::getline(ss, token, ',');
        data.riskFreeRate = std::stod(token);

        std::getline(ss, token, ',');
        data.volatility = std::stod(token);

        std::getline(ss, token, ',');
        data.timeToMaturity = std::stod(token);

        std::getline(ss, token, ',');
        data.optionType = (tolower(token[0]) == 'p')? 1 : 0; // Read first character for 'C' or 'P'

        options.push_back(data);
    }

    file.close();
    return options;
}

struct ThreadData {
    const std::vector<OptionData>* options;
    size_t start;
    size_t end;
    std::vector<float>* results;
};

void* computeOptions(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    const std::vector<OptionData>& options = *(data->options);
    std::vector<float>& results = *(data->results);

    for (size_t i = data->start; i < data->end; ++i) {
        const OptionData& option = options[i];
        results[i] = blackScholes(option.spotPrice, option.strikePrice, option.riskFreeRate,
                                  option.volatility, option.timeToMaturity, option.optionType, 0);
    }
    return nullptr;
}

int main(int argc, char* argv[]) {
    std::string filename = "stocks.csv";
    std::vector<OptionData> options = readCSV(filename);
    size_t numOptions = options.size();

    const size_t numThreads = 4; // Adjust based on your hardware
    size_t chunkSize = (numOptions + numThreads - 1) / numThreads;

    std::vector<pthread_t> threads(numThreads);
    std::vector<ThreadData> threadData(numThreads);
    std::vector<float> results(numOptions, 0.0f);

    // Start measuring time
    auto start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < numThreads; ++i) {
        size_t startIdx = i * chunkSize;
        size_t endIdx = std::min(startIdx + chunkSize, numOptions);

        threadData[i] = {&options, startIdx, endIdx, &results};
        pthread_create(&threads[i], nullptr, computeOptions, &threadData[i]);
    }

    for (size_t i = 0; i < numThreads; ++i) {
        pthread_join(threads[i], nullptr);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Execution Time: " << elapsed.count() << " seconds\n";

    // Uncomment to print results
    // for (size_t i = 0; i < numOptions; ++i) {
    //     std::cout << "Option " << i << ": " << results[i] << "\n";
    // }

    return 0;
}
