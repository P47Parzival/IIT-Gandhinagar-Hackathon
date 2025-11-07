'use client';

import { useState } from 'react';
import { User } from '@supabase/supabase-js';
import { Card, CardContent, CardHeader, CardTitle, Button } from '@/app/components/ui';
import { 
  Database, 
  Server, 
  Key,
  Shield,
  CheckCircle,
  AlertTriangle,
  Settings,
  Plus,
  Edit,
  Trash2,
  Eye,
  EyeOff
} from 'lucide-react';

interface LinkDataDashboardProps {
  user: User;
}

interface SystemConnection {
  id: string;
  name: string;
  type: 'SAP' | 'Oracle' | 'Dynamics' | 'Salesforce' | 'NetSuite' | 'QuickBooks';
  status: 'connected' | 'disconnected' | 'error';
  lastSync: string;
  serverUrl: string;
  database?: string;
}

// Mock existing connections
const mockConnections: SystemConnection[] = [
  {
    id: '1',
    name: 'SAP Production',
    type: 'SAP',
    status: 'disconnected',
    lastSync: '2024-11-05 14:30:00',
    serverUrl: 'sap-prod.company.com',
    database: 'PROD_DB'
  },
  {
    id: '2', 
    name: 'Salesforce CRM',
    type: 'Salesforce',
    status: 'disconnected',
    lastSync: '2024-11-05 14:25:00',
    serverUrl: 'company.salesforce.com'
  },
  {
    id: '3',
    name: 'Oracle Financials',
    type: 'Oracle',
    status: 'disconnected',
    lastSync: '2024-11-04 09:15:00',
    serverUrl: 'oracle-fin.company.com',
    database: 'FINANCE_DB'
  }
];

const systemTypes = [
  { 
    id: 'sap', 
    name: 'SAP', 
    description: 'Connect to SAP ERP, S/4HANA, or Business One',
    icon: '🏢',
    color: 'bg-blue-50 border-blue-200'
  },
  { 
    id: 'oracle', 
    name: 'Oracle', 
    description: 'Oracle ERP Cloud, JD Edwards, or PeopleSoft',
    icon: '🔴',
    color: 'bg-red-50 border-red-200'
  },
  { 
    id: 'dynamics', 
    name: 'Microsoft Dynamics', 
    description: 'Dynamics 365 Finance & Operations or Business Central',
    icon: '🔵',
    color: 'bg-indigo-50 border-indigo-200'
  },
  { 
    id: 'salesforce', 
    name: 'Salesforce', 
    description: 'Salesforce CRM, Revenue Cloud, or Financial Services',
    icon: '☁️',
    color: 'bg-cyan-50 border-cyan-200'
  },
  { 
    id: 'zoho_crm', 
    name: 'Zoho CRM', 
    description: 'Zoho CRM with custom OAuth integration',
    icon: '🎯',
    color: 'bg-purple-50 border-purple-200',
    isCustom: true
  },
  { 
    id: 'netsuite', 
    name: 'NetSuite', 
    description: 'Oracle NetSuite ERP and CRM platform',
    icon: '🟠',
    color: 'bg-orange-50 border-orange-200'
  },
  { 
    id: 'quickbooks', 
    name: 'QuickBooks', 
    description: 'QuickBooks Online or Desktop accounting software',
    icon: '💚',
    color: 'bg-green-50 border-green-200'
  }
];

export function LinkDataDashboard({ user }: LinkDataDashboardProps) {
  const [connections, setConnections] = useState<SystemConnection[]>(mockConnections);
  const [showAddForm, setShowAddForm] = useState(false);
  const [selectedSystemType, setSelectedSystemType] = useState<string>('');
  const [showPassword, setShowPassword] = useState(false);
  const [formData, setFormData] = useState({
    name: '',
    serverUrl: '',
    database: '',
    username: '',
    password: '',
    apiKey: '',
    clientId: '',
    clientSecret: '',
    tenantId: ''
  });

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'connected': return 'text-green-600 bg-green-100';
      case 'error': return 'text-red-600 bg-red-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'connected': return <CheckCircle className="h-4 w-4" />;
      case 'error': return <AlertTriangle className="h-4 w-4" />;
      default: return <Settings className="h-4 w-4" />;
    }
  };

  const handleAddConnection = async () => {
    const systemType = systemTypes.find(s => s.id === selectedSystemType);
    
    // Special handling for Zoho CRM (custom auth)
    if (selectedSystemType === 'zoho_crm') {
      try {
        console.log('Connecting to Zoho CRM with custom auth...');
        
        // First, ensure the custom auth config exists
        const setupResponse = await fetch('/api/authConfig/zoho', {
          method: 'GET',
        });
        
        if (!setupResponse.ok) {
          const setupError = await setupResponse.json();
          if (setupError.setup_required) {
            alert('Zoho CRM integration requires environment variables setup. Please configure ZOHO_CLIENT_ID and ZOHO_CLIENT_SECRET.');
            return;
          }
          throw new Error(setupError.error || 'Failed to setup Zoho CRM auth config');
        }
        
        // Create the connection
        const connectResponse = await fetch('/api/authConfig/zoho', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ action: 'connect' })
        });
        
        if (!connectResponse.ok) {
          const connectError = await connectResponse.json();
          throw new Error(connectError.error || 'Failed to create Zoho CRM connection');
        }
        
        const result = await connectResponse.json();
        
        if (result.redirectUrl) {
          // Open the OAuth URL in a new window
          window.open(result.redirectUrl, '_blank', 'width=600,height=700');
          
          // Add a pending connection to show progress
          const pendingConnection: SystemConnection = {
            id: 'zoho_pending_' + Date.now(),
            name: formData.name || 'Zoho CRM',
            type: 'Salesforce', // Use closest match for type
            status: 'disconnected',
            lastSync: new Date().toISOString().slice(0, 19).replace('T', ' '),
            serverUrl: formData.serverUrl || 'zoho.com'
          };
          
          setConnections([...connections, pendingConnection]);
          
          // Reset form
          setShowAddForm(false);
          setSelectedSystemType('');
          setFormData({
            name: '',
            serverUrl: '',
            database: '',
            username: '',
            password: '',
            apiKey: '',
            clientId: '',
            clientSecret: '',
            tenantId: ''
          });
          
          alert('Zoho CRM OAuth window opened. Please complete the authorization and refresh this page.');
        }
        
      } catch (error) {
        console.error('Error connecting to Zoho CRM:', error);
        alert(`Failed to connect to Zoho CRM: ${error instanceof Error ? error.message : 'Unknown error'}`);
      }
      return;
    }
    
    // Default mock behavior for other systems
    const newConnection: SystemConnection = {
      id: Date.now().toString(),
      name: formData.name,
      type: selectedSystemType as any,
      status: 'connected',
      lastSync: new Date().toISOString().slice(0, 19).replace('T', ' '),
      serverUrl: formData.serverUrl,
      database: formData.database
    };
    
    setConnections([...connections, newConnection]);
    setShowAddForm(false);
    setSelectedSystemType('');
    setFormData({
      name: '',
      serverUrl: '',
      database: '',
      username: '',
      password: '',
      apiKey: '',
      clientId: '',
      clientSecret: '',
      tenantId: ''
    });
  };

  const renderConnectionForm = () => {
    const systemType = systemTypes.find(s => s.id === selectedSystemType);
    if (!systemType) return null;

    return (
      <Card className="bg-white shadow-sm">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Database className="h-5 w-5" />
            Connect to {systemType.name}
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Basic Information */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Connection Name *
              </label>
              <input
                type="text"
                value={formData.name}
                onChange={(e) => setFormData({...formData, name: e.target.value})}
                placeholder={`${systemType.name} Production`}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Server URL *
              </label>
              <input
                type="text"
                value={formData.serverUrl}
                onChange={(e) => setFormData({...formData, serverUrl: e.target.value})}
                placeholder={`${systemType.id}.company.com`}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>
          </div>

          {/* System-specific fields */}
          {(selectedSystemType === 'sap' || selectedSystemType === 'oracle') && (
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Database/Instance
              </label>
              <input
                type="text"
                value={formData.database}
                onChange={(e) => setFormData({...formData, database: e.target.value})}
                placeholder="PROD_DB"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>
          )}

          {/* Authentication */}
          <div className="border-t pt-4">
            <h4 className="text-sm font-medium text-gray-900 mb-3">Authentication</h4>
            
            {selectedSystemType === 'salesforce' ? (
              // OAuth for Salesforce
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Client ID
                  </label>
                  <input
                    type="text"
                    value={formData.clientId}
                    onChange={(e) => setFormData({...formData, clientId: e.target.value})}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Client Secret
                  </label>
                  <div className="relative">
                    <input
                      type={showPassword ? "text" : "password"}
                      value={formData.clientSecret}
                      onChange={(e) => setFormData({...formData, clientSecret: e.target.value})}
                      className="w-full px-3 py-2 pr-10 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    />
                    <button
                      type="button"
                      onClick={() => setShowPassword(!showPassword)}
                      className="absolute right-3 top-1/2 transform -translate-y-1/2"
                    >
                      {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                    </button>
                  </div>
                </div>
              </div>
            ) : (
              // Username/Password for other systems
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Username
                  </label>
                  <input
                    type="text"
                    value={formData.username}
                    onChange={(e) => setFormData({...formData, username: e.target.value})}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Password
                  </label>
                  <div className="relative">
                    <input
                      type={showPassword ? "text" : "password"}
                      value={formData.password}
                      onChange={(e) => setFormData({...formData, password: e.target.value})}
                      className="w-full px-3 py-2 pr-10 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    />
                    <button
                      type="button"
                      onClick={() => setShowPassword(!showPassword)}
                      className="absolute right-3 top-1/2 transform -translate-y-1/2"
                    >
                      {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                    </button>
                  </div>
                </div>
              </div>
            )}

            {/* API Key option for some systems */}
            {(selectedSystemType === 'netsuite' || selectedSystemType === 'quickbooks') && (
              <div className="mt-4">
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  API Key (Optional)
                </label>
                <input
                  type="text"
                  value={formData.apiKey}
                  onChange={(e) => setFormData({...formData, apiKey: e.target.value})}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              </div>
            )}
          </div>

          {/* Action buttons */}
          <div className="flex gap-3 pt-4 border-t">
            <Button 
              onClick={handleAddConnection}
              className="bg-blue-600 hover:bg-blue-700 text-white"
            >
              <Shield className="h-4 w-4 mr-2" />
              Test & Connect
            </Button>
            <Button 
              variant="outline" 
              onClick={() => {
                setShowAddForm(false);
                setSelectedSystemType('');
              }}
            >
              Cancel
            </Button>
          </div>
        </CardContent>
      </Card>
    );
  };

  return (
    <div className="flex-1 overflow-y-auto" style={{ backgroundColor: '#fcfaf9' }}>
      <div className="max-w-6xl mx-auto p-6">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">Link Data Sources</h1>
          <p className="text-gray-600">Connect your ERP, CRM, and financial systems to enable AI-powered insights</p>
        </div>

        {/* Current Connections */}
        <div className="mb-8">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-xl font-semibold text-gray-900">Connected Systems</h2>
            <Button 
              onClick={() => setShowAddForm(true)}
              className="bg-blue-600 hover:bg-blue-700 text-white"
            >
              <Plus className="h-4 w-4 mr-2" />
              Add Connection
            </Button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {connections.map((connection) => (
              <Card key={connection.id} className="bg-white shadow-sm">
                <CardContent className="p-6">
                  <div className="flex items-start justify-between mb-4">
                    <div>
                      <h3 className="font-semibold text-gray-900 mb-1">{connection.name}</h3>
                      <p className="text-sm text-gray-600">{connection.type}</p>
                    </div>
                    <div className="flex gap-2">
                      <button className="text-gray-400 hover:text-gray-600">
                        <Edit className="h-4 w-4" />
                      </button>
                      <button className="text-gray-400 hover:text-red-600">
                        <Trash2 className="h-4 w-4" />
                      </button>
                    </div>
                  </div>
                  
                  <div className={`inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(connection.status)}`}>
                    {getStatusIcon(connection.status)}
                    {connection.status}
                  </div>
                  
                  <div className="mt-4 space-y-2">
                    <div className="flex items-center gap-2 text-sm text-gray-600">
                      <Server className="h-3 w-3" />
                      {connection.serverUrl}
                    </div>
                    {connection.database && (
                      <div className="flex items-center gap-2 text-sm text-gray-600">
                        <Database className="h-3 w-3" />
                        {connection.database}
                      </div>
                    )}
                    <div className="text-xs text-gray-500">
                      Last sync: {new Date(connection.lastSync).toLocaleString()}
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>

        {/* Add Connection Form */}
        {showAddForm && (
          <div className="mb-8">
            {!selectedSystemType ? (
              <Card className="bg-white shadow-sm">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Plus className="h-5 w-5" />
                    Choose System Type
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {systemTypes.map((system) => (
                      <button
                        key={system.id}
                        onClick={() => setSelectedSystemType(system.id)}
                        className={`p-4 rounded-lg border-2 hover:shadow-md transition-all text-left ${system.color}`}
                      >
                        <div className="text-2xl mb-2">{system.icon}</div>
                        <h3 className="font-semibold text-gray-900 mb-1">{system.name}</h3>
                        <p className="text-sm text-gray-600">{system.description}</p>
                      </button>
                    ))}
                  </div>
                </CardContent>
              </Card>
            ) : (
              renderConnectionForm()
            )}
          </div>
        )}

        {/* Footer Note */}
        <div className="text-center text-gray-500 text-sm mt-8">
          <p>🔗 Data linking is currently in preview mode.</p>
        </div>
      </div>
    </div>
  );
}
