'use client';

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle, Button, Badge } from '@/app/components/ui';
import { 
  TrendingUp, 
  FileText, 
  CheckCircle, 
  AlertTriangle,
  BarChart3,
  Calculator
} from 'lucide-react';

interface BalanceSheetData {
  response: string;
  type: string;
  data?: any;
  chart_data?: any;
  suggested_actions?: string[];
}

interface BalanceSheetPanelProps {
  onQuerySubmit: (query: string) => void;
  lastResponse?: BalanceSheetData;
}

export function BalanceSheetPanel({ onQuerySubmit, lastResponse }: BalanceSheetPanelProps) {
  const [suggestions, setSuggestions] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchSuggestions();
  }, []);

  const fetchSuggestions = async () => {
    try {
      const response = await fetch('/api/balance-sheet');
      if (response.ok) {
        const data = await response.json();
        setSuggestions(data.suggestions || []);
      }
    } catch (error) {
      console.error('Failed to fetch suggestions:', error);
    }
  };

  const handleQueryClick = (query: string) => {
    setLoading(true);
    onQuerySubmit(query);
    setTimeout(() => setLoading(false), 2000); // Reset loading after 2 seconds
  };

  const renderKPICards = (data: any) => {
    if (!data || !data.kpi_cards) return null;

    return (
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        {data.kpi_cards.map((kpi: any, index: number) => (
          <Card key={index} className="border-l-4 border-l-blue-500">
            <CardContent className="p-4">
              <div className="text-sm font-medium text-gray-600">{kpi.title}</div>
              <div className="text-2xl font-bold">{kpi.value}</div>
              {kpi.change && (
                <div className={`text-sm ${
                  kpi.trend === 'up' ? 'text-green-600' : 
                  kpi.trend === 'down' ? 'text-red-600' : 'text-gray-600'
                }`}>
                  {kpi.change}
                </div>
              )}
            </CardContent>
          </Card>
        ))}
      </div>
    );
  };

  const renderVarianceData = (data: any) => {
    if (!data || !data.high_variance_accounts) return null;

    return (
      <Card className="mb-6">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <TrendingUp className="h-5 w-5" />
            High Variance Accounts
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {data.high_variance_accounts.slice(0, 5).map((account: any, index: number) => (
              <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <div>
                  <div className="font-medium">{account.account_code}</div>
                  <div className="text-sm text-gray-600">{account.account_name}</div>
                </div>
                <div className="text-right">
                  <div className={`font-medium ${
                    Math.abs(account.variance_percentage) > 50 ? 'text-red-600' :
                    Math.abs(account.variance_percentage) > 20 ? 'text-orange-600' : 'text-yellow-600'
                  }`}>
                    {account.variance_percentage?.toFixed(1)}%
                  </div>
                  <div className="text-sm text-gray-600">
                    ${account.current_amount?.toLocaleString()}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    );
  };

  const renderHygieneScores = (data: any) => {
    if (!data || !data.hygiene_scores) return null;

    return (
      <Card className="mb-6">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <CheckCircle className="h-5 w-5" />
            GL Hygiene Scores
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="mb-4">
            <div className="text-2xl font-bold">{data.average_score?.toFixed(1)}%</div>
            <div className="text-sm text-gray-600">Average Hygiene Score</div>
          </div>
          <div className="space-y-2">
            {data.hygiene_scores.slice(0, 5).map((score: any, index: number) => (
              <div key={index} className="flex items-center justify-between">
                <span className="text-sm">{score.account_code}</span>
                <div className="flex items-center gap-2">
                  <div className="text-sm font-medium">{score.hygiene_score?.toFixed(1)}%</div>
                  <Badge variant={
                    score.grade === 'A' ? 'default' :
                    score.grade === 'B' ? 'secondary' :
                    score.grade === 'C' ? 'outline' : 'destructive'
                  }>
                    {score.grade}
                  </Badge>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    );
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Calculator className="h-6 w-6" />
            Balance Sheet Assurance
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-gray-600 mb-4">
            Ask questions about GL accounts, variances, supporting documents, and compliance status.
          </p>
          
          {/* Quick Actions */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            <Button 
              variant="outline" 
              className="h-auto p-4 flex flex-col items-start gap-2"
              onClick={() => handleQueryClick("Show GLs with variance > 30%")}
              disabled={loading}
            >
              <TrendingUp className="h-5 w-5 text-red-500" />
              <div className="text-left">
                <div className="font-medium">High Variances</div>
                <div className="text-sm text-gray-600">Find accounts with significant variances</div>
              </div>
            </Button>
            
            <Button 
              variant="outline" 
              className="h-auto p-4 flex flex-col items-start gap-2"
              onClick={() => handleQueryClick("List pending supporting documents")}
              disabled={loading}
            >
              <FileText className="h-5 w-5 text-orange-500" />
              <div className="text-left">
                <div className="font-medium">Missing Docs</div>
                <div className="text-sm text-gray-600">Check document upload status</div>
              </div>
            </Button>
            
            <Button 
              variant="outline" 
              className="h-auto p-4 flex flex-col items-start gap-2"
              onClick={() => handleQueryClick("What is the overall hygiene score?")}
              disabled={loading}
            >
              <CheckCircle className="h-5 w-5 text-green-500" />
              <div className="text-left">
                <div className="font-medium">Hygiene Score</div>
                <div className="text-sm text-gray-600">Check GL quality metrics</div>
              </div>
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Response Visualization */}
      {lastResponse && lastResponse.data && (
        <div>
          {renderKPICards(lastResponse.data)}
          {renderVarianceData(lastResponse.data)}
          {renderHygieneScores(lastResponse.data)}
        </div>
      )}

      {/* Suggested Queries */}
      {suggestions.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Suggested Queries</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {suggestions.map((category: any, categoryIndex: number) => (
                <div key={categoryIndex}>
                  <h4 className="font-medium mb-2 flex items-center gap-2">
                    {category.category === 'Variance Analysis' && <TrendingUp className="h-4 w-4" />}
                    {category.category === 'Supporting Documents' && <FileText className="h-4 w-4" />}
                    {category.category === 'Hygiene & Compliance' && <CheckCircle className="h-4 w-4" />}
                    {category.category === 'Account Information' && <BarChart3 className="h-4 w-4" />}
                    {category.category === 'Approvals & Reviews' && <AlertTriangle className="h-4 w-4" />}
                    {category.category}
                  </h4>
                  <div className="flex flex-wrap gap-2">
                    {category.queries?.slice(0, 3).map((query: string, queryIndex: number) => (
                      <Button
                        key={queryIndex}
                        variant="ghost"
                        size="sm"
                        className="text-xs h-auto py-1 px-2"
                        onClick={() => handleQueryClick(query)}
                        disabled={loading}
                      >
                        {query}
                      </Button>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Suggested Actions from Response */}
      {lastResponse?.suggested_actions && lastResponse.suggested_actions.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Recommended Actions</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {lastResponse.suggested_actions.map((action: string, index: number) => (
                <Button
                  key={index}
                  variant="outline"
                  size="sm"
                  className="w-full justify-start"
                  onClick={() => handleQueryClick(action)}
                  disabled={loading}
                >
                  {action}
                </Button>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {loading && (
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-center space-x-2">
              <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
              <span>Processing balance sheet query...</span>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
