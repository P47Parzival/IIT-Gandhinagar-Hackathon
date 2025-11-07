'use client';

import { useState, useEffect } from 'react';
import { User } from '@supabase/supabase-js';
import { Card, CardContent, CardHeader, CardTitle } from '@/app/components/ui';
import { 
  TrendingUp, 
  TrendingDown,
  Users, 
  DollarSign, 
  Clock,
  Star,
  BarChart3,
  PieChart,
  Target
} from 'lucide-react';

interface AnalyticsDashboardProps {
  user: User;
}

// Mock financial/CRM analytics data
const mockFinancialKPIs = {
  totalGLVariance: { value: 2.34, label: 'Total GL Variance (%)', trend: 'down', change: -0.12 },
  reconciliationRate: { value: 94.7, label: 'Reconciliation Completion (%)', trend: 'up', change: 2.3 },
  balanceSheetHealth: { value: 87, label: 'Balance Sheet Health Score', trend: 'up', change: 3 },
  outstandingItems: { value: 156, label: 'Outstanding Reconciliation Items', trend: 'down', change: -23 },
  monthlyCloseEfficiency: { value: 6.2, label: 'Monthly Close Days', trend: 'down', change: -0.8 }
};

const mockTopAccountVariances = [
  { account: 'Cash & Cash Equivalents', code: '1001', variance: 4.23, amount: 125000 },
  { account: 'Accounts Receivable', code: '1200', variance: 3.85, amount: 89000 },
  { account: 'Inventory', code: '1300', variance: 2.94, amount: 67000 },
  { account: 'Fixed Assets', code: '1500', variance: 2.11, amount: 45000 },
  { account: 'Accounts Payable', code: '2100', variance: 1.87, amount: 38000 }
];

const mockAssetsVsLiabilities = [
  { category: 'Current Assets', amount: 2450000, percentage: 45.2, color: '#10b981' },
  { category: 'Fixed Assets', amount: 1890000, percentage: 34.8, color: '#1f2937' },
  { category: 'Current Liabilities', amount: 750000, percentage: 13.8, color: '#ef4444' },
  { category: 'Long-term Liabilities', amount: 340000, percentage: 6.2, color: '#f59e0b' }
];

const mockBalanceSheetComposition = [
  { category: 'Cash & Equivalents', percentage: 18.5, color: '#3b82f6' },
  { category: 'Accounts Receivable', percentage: 22.1, color: '#10b981' },
  { category: 'Inventory', percentage: 15.8, color: '#f59e0b' },
  { category: 'Fixed Assets', percentage: 28.3, color: '#1f2937' },
  { category: 'Other Assets', percentage: 15.3, color: '#8b5cf6' }
];

const mockGLVarianceTrend = [
  { month: 'Jan', variance: 1.85 },
  { month: 'Feb', variance: 2.12 },
  { month: 'Mar', variance: 1.94 },
  { month: 'Apr', variance: 2.48 },
  { month: 'May', variance: 2.87 },
  { month: 'Jun', variance: 2.34 },
  { month: 'Jul', variance: 1.98 }
];

const mockDepartmentFinancials = [
  { department: 'Sales', glAccuracy: 96.2, variance: 1.2 },
  { department: 'Marketing', glAccuracy: 94.8, variance: 1.8 },
  { department: 'Operations', glAccuracy: 92.5, variance: 2.4 },
  { department: 'Technology', glAccuracy: 89.1, variance: 3.1 },
  { department: 'Finance', glAccuracy: 98.7, variance: 0.8 },
  { department: 'HR', glAccuracy: 91.3, variance: 2.9 }
];

export function AnalyticsDashboard({ user }: AnalyticsDashboardProps) {
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Simulate loading
    const timer = setTimeout(() => {
      setLoading(false);
    }, 1000);

    return () => clearTimeout(timer);
  }, []);

  if (loading) {
    return (
      <div className="flex-1 flex items-center justify-center" style={{ backgroundColor: '#fcfaf9' }}>
        <div className="text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-gray-900 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading analytics dashboard...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto" style={{ backgroundColor: '#fcfaf9' }}>
      <div className="max-w-7xl mx-auto p-6">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">Financial Analytics Dashboard</h1>
          <p className="text-gray-600">General Ledger & Balance Sheet Performance Overview</p>
        </div>

        {/* Financial KPI Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-6 mb-8">
          <Card className="bg-white shadow-sm">
            <CardContent className="p-6">
              <div className="flex items-center">
                <TrendingDown className="h-8 w-8 text-red-600" />
                <div className="ml-4">
                  <p className="text-sm font-medium text-gray-600">{mockFinancialKPIs.totalGLVariance.label}</p>
                  <div className="flex items-center">
                    <p className="text-2xl font-bold text-gray-900">{mockFinancialKPIs.totalGLVariance.value}%</p>
                    <span className="ml-2 text-sm text-green-600">({mockFinancialKPIs.totalGLVariance.change}%)</span>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-white shadow-sm">
            <CardContent className="p-6">
              <div className="flex items-center">
                <Target className="h-8 w-8 text-green-600" />
                <div className="ml-4">
                  <p className="text-sm font-medium text-gray-600">{mockFinancialKPIs.reconciliationRate.label}</p>
                  <div className="flex items-center">
                    <p className="text-2xl font-bold text-gray-900">{mockFinancialKPIs.reconciliationRate.value}%</p>
                    <span className="ml-2 text-sm text-green-600">(+{mockFinancialKPIs.reconciliationRate.change}%)</span>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-white shadow-sm">
            <CardContent className="p-6">
              <div className="flex items-center">
                <Star className="h-8 w-8 text-yellow-600" />
                <div className="ml-4">
                  <p className="text-sm font-medium text-gray-600">{mockFinancialKPIs.balanceSheetHealth.label}</p>
                  <div className="flex items-center">
                    <p className="text-2xl font-bold text-gray-900">{mockFinancialKPIs.balanceSheetHealth.value}</p>
                    <span className="ml-2 text-sm text-green-600">(+{mockFinancialKPIs.balanceSheetHealth.change})</span>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-white shadow-sm">
            <CardContent className="p-6">
              <div className="flex items-center">
                <Users className="h-8 w-8 text-orange-600" />
                <div className="ml-4">
                  <p className="text-sm font-medium text-gray-600">{mockFinancialKPIs.outstandingItems.label}</p>
                  <div className="flex items-center">
                    <p className="text-2xl font-bold text-gray-900">{mockFinancialKPIs.outstandingItems.value}</p>
                    <span className="ml-2 text-sm text-green-600">({mockFinancialKPIs.outstandingItems.change})</span>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-white shadow-sm">
            <CardContent className="p-6">
              <div className="flex items-center">
                <Clock className="h-8 w-8 text-blue-600" />
                <div className="ml-4">
                  <p className="text-sm font-medium text-gray-600">{mockFinancialKPIs.monthlyCloseEfficiency.label}</p>
                  <div className="flex items-center">
                    <p className="text-2xl font-bold text-gray-900">{mockFinancialKPIs.monthlyCloseEfficiency.value}</p>
                    <span className="ml-2 text-sm text-green-600">({mockFinancialKPIs.monthlyCloseEfficiency.change})</span>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Main Question */}
        <div className="mb-6">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">Is the GL Variance within acceptable limits?</h2>
        </div>

        {/* Charts Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6 mb-8">
          {/* Top Account Variances Bar Chart */}
          <Card className="bg-white shadow-sm">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <BarChart3 className="h-5 w-5" />
                Top Account Variances
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {mockTopAccountVariances.map((account, index) => (
                  <div key={account.code} className="flex items-center">
                    <div className="w-16 text-xs text-gray-600">
                      <div className="font-medium">{account.code}</div>
                    </div>
                    <div className="flex-1 mx-4">
                      <div className="bg-gray-200 rounded-full h-6 relative">
                        <div 
                          className="bg-red-500 h-6 rounded-full flex items-center justify-end pr-2"
                          style={{ width: `${(account.variance / mockTopAccountVariances[0].variance) * 100}%` }}
                        >
                          <span className="text-white text-xs font-medium">{account.variance}%</span>
                        </div>
                      </div>
                      <div className="text-xs text-gray-500 mt-1 truncate">{account.account}</div>
                    </div>
                    <div className="text-xs text-gray-600 w-16 text-right">
                      ${(account.amount / 1000).toFixed(0)}K
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Assets vs Liabilities Pie Chart */}
          <Card className="bg-white shadow-sm">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <PieChart className="h-5 w-5" />
                Assets vs Liabilities
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="relative h-48 mb-4">
                {/* Mock pie chart visualization */}
                <div className="w-32 h-32 mx-auto rounded-full relative" style={{
                  background: `conic-gradient(
                    #10b981 0% ${mockAssetsVsLiabilities[0].percentage}%,
                    #1f2937 ${mockAssetsVsLiabilities[0].percentage}% ${mockAssetsVsLiabilities[0].percentage + mockAssetsVsLiabilities[1].percentage}%,
                    #ef4444 ${mockAssetsVsLiabilities[0].percentage + mockAssetsVsLiabilities[1].percentage}% ${mockAssetsVsLiabilities[0].percentage + mockAssetsVsLiabilities[1].percentage + mockAssetsVsLiabilities[2].percentage}%,
                    #f59e0b ${mockAssetsVsLiabilities[0].percentage + mockAssetsVsLiabilities[1].percentage + mockAssetsVsLiabilities[2].percentage}% 100%
                  )`
                }}>
                  <div className="absolute inset-4 bg-white rounded-full flex items-center justify-center">
                    <span className="text-xs font-semibold text-gray-700">$5.43M</span>
                  </div>
                </div>
              </div>
              <div className="space-y-2">
                {mockAssetsVsLiabilities.map((item, index) => (
                  <div key={item.category} className="flex items-center justify-between">
                    <div className="flex items-center">
                      <div 
                        className="w-3 h-3 rounded-full mr-2" 
                        style={{ backgroundColor: item.color }}
                      ></div>
                      <span className="text-sm text-gray-600">{item.category}</span>
                    </div>
                    <span className="text-sm font-medium">${(item.amount / 1000000).toFixed(1)}M</span>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Balance Sheet Composition */}
          <Card className="bg-white shadow-sm">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <PieChart className="h-5 w-5" />
                Balance Sheet Composition
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="relative h-48 mb-4">
                {/* Mock pie chart visualization */}
                <div className="w-32 h-32 mx-auto rounded-full relative" style={{
                  background: `conic-gradient(
                    #3b82f6 0% 18.5%,
                    #10b981 18.5% 40.6%,
                    #f59e0b 40.6% 56.4%,
                    #1f2937 56.4% 84.7%,
                    #8b5cf6 84.7% 100%
                  )`
                }}>
                  <div className="absolute inset-4 bg-white rounded-full flex items-center justify-center">
                    <span className="text-xs font-semibold text-gray-700">Assets</span>
                  </div>
                </div>
              </div>
              <div className="space-y-2">
                {mockBalanceSheetComposition.map((item, index) => (
                  <div key={item.category} className="flex items-center justify-between">
                    <div className="flex items-center">
                      <div 
                        className="w-3 h-3 rounded-full mr-2" 
                        style={{ backgroundColor: item.color }}
                      ></div>
                      <span className="text-sm text-gray-600">{item.category}</span>
                    </div>
                    <span className="text-sm font-medium">{item.percentage}%</span>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* GL Performance Section */}
        <div className="mb-6">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">How is the monthly GL performance trending?</h2>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          {/* Monthly GL Variance Trend */}
          <Card className="bg-white shadow-sm">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <TrendingUp className="h-5 w-5" />
                Monthly GL Variance Trend
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {mockGLVarianceTrend.map((item, index) => (
                  <div key={item.month} className="flex items-center">
                    <div className="w-12 text-sm text-gray-600 font-medium">{item.month}</div>
                    <div className="flex-1 mx-4">
                      <div className="bg-gray-200 rounded-full h-6 relative">
                        <div 
                          className={`h-6 rounded-full flex items-center justify-end pr-2 ${
                            item.variance > 2.5 ? 'bg-red-500' : 
                            item.variance > 2.0 ? 'bg-yellow-500' : 'bg-green-500'
                          }`}
                          style={{ width: `${(item.variance / Math.max(...mockGLVarianceTrend.map(d => d.variance))) * 100}%` }}
                        >
                          <span className="text-white text-xs font-medium">{item.variance}%</span>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Department GL Performance */}
          <Card className="bg-white shadow-sm">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Target className="h-5 w-5" />
                Department GL Accuracy & Variance
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {mockDepartmentFinancials.map((dept, index) => (
                  <div key={dept.department} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                    <span className="text-sm font-medium text-gray-700">{dept.department}</span>
                    <div className="flex items-center gap-4">
                      <div className="flex items-center">
                        <div className="w-16 bg-gray-200 rounded-full h-2 mr-2">
                          <div 
                            className={`h-2 rounded-full ${
                              dept.glAccuracy > 95 ? 'bg-green-500' :
                              dept.glAccuracy > 90 ? 'bg-yellow-500' : 'bg-red-500'
                            }`}
                            style={{ width: `${dept.glAccuracy}%` }}
                          ></div>
                        </div>
                        <span className="text-xs font-bold text-gray-900 w-10">{dept.glAccuracy}%</span>
                      </div>
                      <div className="text-xs text-gray-600">
                        Var: <span className={`font-medium ${
                          dept.variance < 1.5 ? 'text-green-600' :
                          dept.variance < 2.5 ? 'text-yellow-600' : 'text-red-600'
                        }`}>{dept.variance}%</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Footer Note */}
        <div className="text-center text-gray-500 text-sm">
          <p>📊 Financial dashboard data is currently mocked. Integration with SAP/CRM servers coming soon.</p>
        </div>
      </div>
    </div>
  );
}

