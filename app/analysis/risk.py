"""
Risk analysis - Risk scoring, entry/TP/DCA levels, position sizing
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from app.config import ML_AVAILABLE


def calculate_risk_score(features: Dict, metrics: Dict, info: Dict) -> Dict:
    """
    Calculate risk score (0-100) using ensemble of risk factors
    
    Args:
        features: Dict with ML features
        metrics: Dict with calculated metrics
        info: Stock info dict
        
    Returns:
        Dict with risk score, category, and factors
    """
    risk_factors = {}
    risk_score = 0.0
    
    # Volatility Risk (0-25 points)
    volatility = features.get('volatility', 30)
    if volatility > 60:
        vol_risk = 25
    elif volatility > 40:
        vol_risk = 18
    elif volatility > 25:
        vol_risk = 10
    else:
        vol_risk = 5
    risk_factors['volatility'] = vol_risk
    risk_score += vol_risk
    
    # Beta Risk (0-15 points)
    beta = features.get('beta', 1.0)
    if beta > 1.5:
        beta_risk = 15
    elif beta > 1.2:
        beta_risk = 10
    elif beta < 0.8:
        beta_risk = 5
    else:
        beta_risk = 8
    risk_factors['market_correlation'] = beta_risk
    risk_score += beta_risk
    
    # Financial Health Risk (0-20 points)
    debt_to_equity = features.get('debt_to_equity', 0.5)
    current_ratio = features.get('current_ratio', 1.5)
    roe = features.get('roe', 10)
    
    financial_risk = 0
    if debt_to_equity > 2.0:
        financial_risk += 10
    elif debt_to_equity > 1.0:
        financial_risk += 5
    
    if current_ratio < 1.0:
        financial_risk += 8
    elif current_ratio < 1.5:
        financial_risk += 4
    
    if roe < 0:
        financial_risk += 7
    
    risk_factors['financial_health'] = min(20, financial_risk)
    risk_score += min(20, financial_risk)
    
    # Liquidity Risk (0-15 points)
    market_cap_category = features.get('market_cap_category', 2.0)
    volume_ratio = features.get('volume_ratio', 1.0)
    
    liquidity_risk = 0
    if market_cap_category == 1.0:  # Small cap
        liquidity_risk += 8
    elif market_cap_category == 2.0:  # Mid cap
        liquidity_risk += 4
    
    if volume_ratio < 0.5:  # Low volume
        liquidity_risk += 7
    
    risk_factors['liquidity'] = min(15, liquidity_risk)
    risk_score += min(15, liquidity_risk)
    
    # Sector/Industry Risk (0-10 points)
    sector_risk = 5  # Default medium risk
    risk_factors['sector_risk'] = sector_risk
    risk_score += sector_risk
    
    # Concentration Risk (0-15 points)
    concentration_risk = 5  # Default medium risk
    risk_factors['concentration'] = concentration_risk
    risk_score += concentration_risk
    
    # Normalize to 0-100
    risk_score = min(100, max(0, risk_score))
    
    # Determine risk category
    if risk_score < 25:
        risk_category = 'Low'
    elif risk_score < 50:
        risk_category = 'Medium'
    elif risk_score < 75:
        risk_category = 'High'
    else:
        risk_category = 'Very High'
    
    # Key risk factors (top 3)
    sorted_risks = sorted(risk_factors.items(), key=lambda x: x[1], reverse=True)
    key_risk_factors = [{'factor': k, 'score': v} for k, v in sorted_risks[:3]]
    
    return {
        'risk_score': round(risk_score, 1),
        'risk_category': risk_category,
        'risk_factors': risk_factors,
        'key_risk_factors': key_risk_factors,
        'model_type': 'ensemble'
    }


# Functions will be moved here from app.py:
# - calculate_entry_tp_dca()
# - calculate_position_sizing()

