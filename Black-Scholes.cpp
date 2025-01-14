#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <iostream>

#define inv_sqrt_2xPI 0.39894228040143270286

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

int main(int argc, char *argv[])
{
    float spotprice ;
    float strike    ;
    float rate      ;
    float volatility;
    float otime     ;
    int   otype     ;
    char  otype_c   ;

    printf("Spot Price = "        ); scanf("%f", &spotprice );
    printf("Strike Price = "      ); scanf("%f", &strike    );
    printf("Risk-Free Rate = "    ); scanf("%f", &rate      );
    printf("Volatility = "        ); scanf("%f", &volatility);
    printf("Time-to-Maturity = "  ); scanf("%f", &otime     );
    printf("Option Type "         );
    printf("('P:put, 'C':call) = "); scanf("\n%c", &otype_c    );

    otype = (tolower(otype_c) == 'p')? 1 : 0;

    printf("\n\n");
    printf("Spot Price:       %f \n", spotprice);
    printf("Strike Price:     %f \n", strike);
    printf("Risk-Free Rate:   %f \n", rate);
    printf("Volatility:       %f \n", volatility);
    printf("Time-to-Maturity: %f \n", otime);
    printf("Option Type:      %s  \n", (otype == 1) ? "put" : "call");
    
    float oprice = blackScholes(spotprice, strike, rate, volatility, otime, otype, 0);
    
    printf("Option Price:     %f \n", oprice);
    
    return 0;
}