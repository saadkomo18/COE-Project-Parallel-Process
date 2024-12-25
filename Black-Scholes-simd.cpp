#include <immintrin.h>
#include <cmath>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono> // For measuring execution time
// Struct to hold option data
struct OptionData {
    float spotPrice;
    float strikePrice;
    float riskFreeRate;
    float volatility;
    float timeToMaturity;
    char optionType; // 'C' for Call, 'P' for Put
};

// Function to read CSV file and split data into call and put streams
void processCSV(const std::string& filename, std::vector<OptionData>& callOptions, std::vector<OptionData>& putOptions) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }

    std::string line;
    bool isHeader = true;

    while (std::getline(file, line)) {
        if (isHeader) {
            isHeader = false; // Skip the header line
            continue;
        }

        std::stringstream ss(line);
        std::string token;
        OptionData data;

        std::getline(ss, token, ',');
        data.spotPrice = std::stof(token);

        std::getline(ss, token, ',');
        data.strikePrice = std::stof(token);

        std::getline(ss, token, ',');
        data.riskFreeRate = std::stof(token);

        std::getline(ss, token, ',');
        data.volatility = std::stof(token);

        std::getline(ss, token, ',');
        data.timeToMaturity = std::stof(token);

        std::getline(ss, token, ',');
        data.optionType = token[0]; // 'C' for Call, 'P' for Put

        if (data.optionType == 'C') {
            callOptions.push_back(data);
        } else if (data.optionType == 'P') {
            putOptions.push_back(data);
        }
    }
}

// Helper function to compute the cumulative normal distribution function
__m256 CNDF(__m256 d) {
    /*_mm256_set1_ps:
    Broadcast single-precision (32-bit) floating-point value a to all elements of dst.*/
    const __m256 sqrt2 = _mm256_set1_ps(1.41421356237f); // √2
    const __m256 half = _mm256_set1_ps(0.5f);

    // Compute x / √2
    /*_mm256_div_ps:
    Divide packed single-precision (32-bit) floating-point elements in a by packed elements in b, and store the results in dst.*/
    __m256 xDivSqrt2 = _mm256_div_ps(d, sqrt2);

    // Scalar compute erf for each element
    float temp[8];
    /*_mm256_storeu_ps:
    Store 256-bits (composed of 8 packed single-precision (32-bit) floating-point elements) into memory. mem_addr does not need to be aligned on any particular boundary.*/
    _mm256_storeu_ps(temp, xDivSqrt2);
    for (int i = 0; i < 8; ++i) {
        temp[i] = std::erf(temp[i]);
    }

    // Reload into AVX
    /*_mm256_loadu_ps:
    Load 256-bits (composed of 8 packed single-precision (32-bit) floating-point elements) from memory into dst. mem_addr does not need to be aligned on any particular boundary.*/
    __m256 erfResult = _mm256_loadu_ps(temp);

    // Compute N(d) = 0.5 * (1 + erf(x / √2))
    /*_mm256_mul_ps:
    Multiply packed single-precision (32-bit) floating-point elements in a and b, and store the results in dst.
    _mm256_add_ps:
    Add packed single-precision (32-bit) floating-point elements in a and b, and store the results in dst.
    _mm256_set1_ps:
    Broadcast single-precision (32-bit) floating-point value a to all elements of dst.*/
    return _mm256_mul_ps(half, _mm256_add_ps(_mm256_set1_ps(1.0f), erfResult));
}

// Black-Scholes Call Option Calculation
void blackScholesCall(const std::vector<OptionData>& callOptions, std::vector<float>& prices) {
    prices.resize(callOptions.size()); // Ensure prices vector is large enough
    // the variables bellow: one and half, were declared to be used in some avx commands
    /*_mm256_set1_ps:
    Broadcast single-precision (32-bit) floating-point value a to all elements of dst.*/
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 half = _mm256_set1_ps(0.5f);

    size_t count = callOptions.size();

    for (size_t i = 0; i < count; i += 8) {
        // Load 8 inputs into AVX registers
        /*_mm256_loadu_ps: 
        Set packed single-precision (32-bit) floating-point elements in dst with the supplied values.*/
        __m256 S = _mm256_set_ps(
            callOptions[i + 7].spotPrice, callOptions[i + 6].spotPrice, callOptions[i + 5].spotPrice, callOptions[i + 4].spotPrice,
            callOptions[i + 3].spotPrice, callOptions[i + 2].spotPrice, callOptions[i + 1].spotPrice, callOptions[i].spotPrice);
        __m256 K = _mm256_set_ps(
            callOptions[i + 7].strikePrice, callOptions[i + 6].strikePrice, callOptions[i + 5].strikePrice, callOptions[i + 4].strikePrice,
            callOptions[i + 3].strikePrice, callOptions[i + 2].strikePrice, callOptions[i + 1].strikePrice, callOptions[i].strikePrice);
        __m256 r = _mm256_set_ps(
            callOptions[i + 7].riskFreeRate, callOptions[i + 6].riskFreeRate, callOptions[i + 5].riskFreeRate, callOptions[i + 4].riskFreeRate,
            callOptions[i + 3].riskFreeRate, callOptions[i + 2].riskFreeRate, callOptions[i + 1].riskFreeRate, callOptions[i].riskFreeRate);
        __m256 sigma = _mm256_set_ps(
            callOptions[i + 7].volatility, callOptions[i + 6].volatility, callOptions[i + 5].volatility, callOptions[i + 4].volatility,
            callOptions[i + 3].volatility, callOptions[i + 2].volatility, callOptions[i + 1].volatility, callOptions[i].volatility);
        __m256 T = _mm256_set_ps(
            callOptions[i + 7].timeToMaturity, callOptions[i + 6].timeToMaturity, callOptions[i + 5].timeToMaturity, callOptions[i + 4].timeToMaturity,
            callOptions[i + 3].timeToMaturity, callOptions[i + 2].timeToMaturity, callOptions[i + 1].timeToMaturity, callOptions[i].timeToMaturity);

        // Compute scalar log(S / K) manually, avx does not support log, so it should be using arrays 
        float logValues[8];
        for (int j = 0; j < 8 && (i + j) < count; ++j) {
            logValues[j] = std::log(callOptions[i + j].spotPrice / callOptions[i + j].strikePrice);
        }
        /*_mm256_loadu_ps: 
        Load 256-bits (composed of 8 packed single-precision (32-bit) floating-point elements) from 
        memory into dst. mem_addr does not need to be aligned on any particular boundary.*/
        __m256 logSK = _mm256_loadu_ps(logValues);

        // Compute d1 and d2
        /* _mm256_mul_ps:
        Multiply packed single-precision (32-bit) floating-point elements in a and b, and store the results in dst.*/
        __m256 sigma2Half = _mm256_mul_ps(half, _mm256_mul_ps(sigma, sigma)); // 0.5 * sigma^2
        /*_mm256_add_ps:
        Add packed single-precision (32-bit) floating-point elements in a and b, and store the results in dst.*/
        __m256 numerator = _mm256_add_ps(logSK, _mm256_mul_ps(_mm256_add_ps(r, sigma2Half), T));
        /*_mm256_sqrt_ps: 
        Compute the square root of packed single-precision (32-bit) floating-point elements in a, and store the results in dst.*/
        __m256 denominator = _mm256_mul_ps(sigma, _mm256_sqrt_ps(T));
        /*_mm256_div_ps:
        Divide packed single-precision (32-bit) floating-point elements in a by packed elements in b, and store the results in dst.*/
        __m256 d1 = _mm256_div_ps(numerator, denominator);
        /*_mm256_sub_ps:
        Subtract packed single-precision (32-bit) floating-point elements in b from packed single-precision (32-bit) floating-point elements in a, and store the results in dst.*/
        __m256 d2 = _mm256_sub_ps(d1, denominator);

        // Compute N(d1) and N(d2)
        __m256 Nd1 = CNDF(d1);
        __m256 Nd2 = CNDF(d2);

        // Compute scalar exp(-rT) manually, because avx does not support exp
        float expValues[8];
        for (int j = 0; j < 8 && (i + j) < count; ++j) {
            expValues[j] = std::exp(-callOptions[i + j].riskFreeRate * callOptions[i + j].timeToMaturity);
        }
        /*_mm256_loadu_ps:
        Load 256-bits (composed of 8 packed single-precision (32-bit) floating-point elements) from memory into dst. mem_addr does not need to be aligned on any particular boundary.*/
        __m256 expRT = _mm256_loadu_ps(expValues);

        /* _mm256_mul_ps:
        Multiply packed single-precision (32-bit) floating-point elements in a and b, and store the results in dst.*/
        __m256 discountedK = _mm256_mul_ps(K, expRT);

        // Compute Call Price: C = S N(d1) - K e^(-rT) N(d2)
        /*_mm256_sub_ps:
        Subtract packed single-precision (32-bit) floating-point elements in b from packed single-precision (32-bit) floating-point elements in a, and store the results in dst.*/
        /* _mm256_mul_ps:
        Multiply packed single-precision (32-bit) floating-point elements in a and b, and store the results in dst.*/
        __m256 callPrice = _mm256_sub_ps(_mm256_mul_ps(S, Nd1), _mm256_mul_ps(discountedK, Nd2));

        // Store results in the prices vector
        for (int j = 0; j < 8 && (i + j) < count; ++j) {
            prices[i + j] = ((float*)&callPrice)[j];
        }
    }
}

// Black-Scholes Put Option Calculation
void blackScholesPut(const std::vector<OptionData>& putOptions, std::vector<float>& prices) {
    prices.resize(putOptions.size()); // Ensure prices vector is large enough
    // the variables bellow: one and half, were declared to be used in some avx commands
    /*_mm256_set1_ps:
    Broadcast single-precision (32-bit) floating-point value a to all elements of dst.*/
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 half = _mm256_set1_ps(0.5f);

    size_t count = putOptions.size();

    for (size_t i = 0; i < count; i += 8) {
        // Load 8 inputs into AVX registers
        /*_mm256_loadu_ps: 
        Set packed single-precision (32-bit) floating-point elements in dst with the supplied values.*/
        __m256 S = _mm256_set_ps(
            putOptions[i + 7].spotPrice, putOptions[i + 6].spotPrice, putOptions[i + 5].spotPrice, putOptions[i + 4].spotPrice,
            putOptions[i + 3].spotPrice, putOptions[i + 2].spotPrice, putOptions[i + 1].spotPrice, putOptions[i].spotPrice);
        __m256 K = _mm256_set_ps(
            putOptions[i + 7].strikePrice, putOptions[i + 6].strikePrice, putOptions[i + 5].strikePrice, putOptions[i + 4].strikePrice,
            putOptions[i + 3].strikePrice, putOptions[i + 2].strikePrice, putOptions[i + 1].strikePrice, putOptions[i].strikePrice);
        __m256 r = _mm256_set_ps(
            putOptions[i + 7].riskFreeRate, putOptions[i + 6].riskFreeRate, putOptions[i + 5].riskFreeRate, putOptions[i + 4].riskFreeRate,
            putOptions[i + 3].riskFreeRate, putOptions[i + 2].riskFreeRate, putOptions[i + 1].riskFreeRate, putOptions[i].riskFreeRate);
        __m256 sigma = _mm256_set_ps(
            putOptions[i + 7].volatility, putOptions[i + 6].volatility, putOptions[i + 5].volatility, putOptions[i + 4].volatility,
            putOptions[i + 3].volatility, putOptions[i + 2].volatility, putOptions[i + 1].volatility, putOptions[i].volatility);
        __m256 T = _mm256_set_ps(
            putOptions[i + 7].timeToMaturity, putOptions[i + 6].timeToMaturity, putOptions[i + 5].timeToMaturity, putOptions[i + 4].timeToMaturity,
            putOptions[i + 3].timeToMaturity, putOptions[i + 2].timeToMaturity, putOptions[i + 1].timeToMaturity, putOptions[i].timeToMaturity);

        // Compute scalar log(S / K) manually, avx does not support log, so it should be done using arrays 
        float logValues[8];
        for (int j = 0; j < 8 && (i + j) < count; ++j) {
            logValues[j] = std::log(putOptions[i + j].spotPrice / putOptions[i + j].strikePrice);
        }
         /*_mm256_loadu_ps: 
        Load 256-bits (composed of 8 packed single-precision (32-bit) floating-point elements) from 
        memory into dst. mem_addr does not need to be aligned on any particular boundary.*/
        __m256 logSK = _mm256_loadu_ps(logValues);

        // Compute d1 and d2
        /* _mm256_mul_ps:
        Multiply packed single-precision (32-bit) floating-point elements in a and b, and store the results in dst.*/ 
        __m256 sigma2Half = _mm256_mul_ps(half, _mm256_mul_ps(sigma, sigma)); // 0.5 * sigma^2
        /*_mm256_add_ps:
        Add packed single-precision (32-bit) floating-point elements in a and b, and store the results in dst.*/
        __m256 numerator = _mm256_add_ps(logSK, _mm256_mul_ps(_mm256_add_ps(r, sigma2Half), T));
        /*_mm256_sqrt_ps: 
        Compute the square root of packed single-precision (32-bit) floating-point elements in a, and store the results in dst.*/
        __m256 denominator = _mm256_mul_ps(sigma, _mm256_sqrt_ps(T));
         /*_mm256_div_ps:
        Divide packed single-precision (32-bit) floating-point elements in a by packed elements in b, and store the results in dst.*/
        __m256 d1 = _mm256_div_ps(numerator, denominator);
        /*_mm256_sub_ps:
        Subtract packed single-precision (32-bit) floating-point elements in b from packed single-precision (32-bit) floating-point elements in a, and store the results in dst.*/
        __m256 d2 = _mm256_sub_ps(d1, denominator);

        // Compute N(-d1) and N(-d2), to achive that we will subtract 0 from add d1 and d2 variables before passing them to CNDF.
        /*_mm256_setzero_ps:
        Return vector of type __m256 with all elements set to zero.*/
        __m256 Nd1 = CNDF(_mm256_sub_ps(_mm256_setzero_ps(), d1));
        __m256 Nd2 = CNDF(_mm256_sub_ps(_mm256_setzero_ps(), d2));

        // Compute scalar exp(-rT) manually, because avx does not support exp
        float expValues[8];
        for (int j = 0; j < 8 && (i + j) < count; ++j) {
            expValues[j] = std::exp(-putOptions[i + j].riskFreeRate * putOptions[i + j].timeToMaturity);
        }
        __m256 expRT = _mm256_loadu_ps(expValues);
        /* _mm256_mul_ps:
        Multiply packed single-precision (32-bit) floating-point elements in a and b, and store the results in dst.*/
        __m256 discountedK = _mm256_mul_ps(K, expRT);

        // Compute Put Price: P = K e^(-rT) N(-d2) - S N(-d1)
        __m256 putPrice = _mm256_sub_ps(_mm256_mul_ps(discountedK, Nd2), _mm256_mul_ps(S, Nd1));

        // Store results in the prices vector
        for (int j = 0; j < 8 && (i + j) < count; ++j) {
            prices[i + j] = ((float*)&putPrice)[j];
        }
    }
}

int main() {
    // Prepare the vectors that will hold the data from the input file:
    // This vector will hold all call recods
    std::vector<OptionData> callOptions;
    // This vector will hold all put recods
    std::vector<OptionData> putOptions;

    // This function will fill the above vectors with appropriate data
    processCSV("stocks.csv", callOptions, putOptions);

    // Vectors to store computed call and put prices
    std::vector<float> callPrices;
    std::vector<float> putPrices;
    // Start calculating the execution time
    auto start = std::chrono::high_resolution_clock::now();

    // Compute call and put option prices
    blackScholesCall(callOptions, callPrices);
    blackScholesPut(putOptions, putPrices);
    
    //Stop calculating the execution time
    auto end = std::chrono::high_resolution_clock::now();

    //Print the results
    /*
    std::cout << "Call Option Prices:\n";
    for (size_t i = 0; i < callPrices.size(); ++i) {
        std::cout << "Spot price: " << callOptions[i].spotPrice << "\n";
        std::cout << "Strike price: " << callOptions[i].strikePrice << "\n";
        std::cout << "Risk Free Rate: " <<  callOptions[i].riskFreeRate << "\n";
        std::cout << "Volatility: " << callOptions[i].volatility << "\n";
        std::cout << "Time To Maturity: " << callOptions[i].timeToMaturity << "\n";
        std::cout << "Option Type: " << callOptions[i].optionType << "\n";
        std::cout << "Option Price: " << callPrices[i] << "\n";
    }

    std::cout << "Call Option Prices:\n";
    for (size_t i = 0; i < callPrices.size(); ++i) {
        std::cout << "Spot price: " << putOptions[i].spotPrice << "\n";
        std::cout << "Strike price: " << putOptions[i].strikePrice << "\n";
        std::cout << "Risk Free Rate: " <<  putOptions[i].riskFreeRate << "\n";
        std::cout << "Volatility: " << putOptions[i].volatility << "\n";
        std::cout << "Time To Maturity: " << putOptions[i].timeToMaturity << "\n";
        std::cout << "Option Type: " << putOptions[i].optionType << "\n";
        std::cout << "Option Price: " << putPrices[i] << "\n";
    }
    */
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Execution Time: " << elapsed.count() << " seconds\n";
    return 0;
}

