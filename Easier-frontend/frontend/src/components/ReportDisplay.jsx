import React, { useState, useEffect } from 'react';
import { 
  ChevronDown, 
  Calendar, 
  Tag, 
  CheckCircle,
  Clock,
  AlertCircle,
  XCircle,
  Inbox,
  Filter,
  Search,
  Download,
  Eye,
  MoreHorizontal,
  Trash2,
  Edit3,
  Share2,
  Star,
  TrendingUp,
  Users,
  Activity
} from 'lucide-react';

function ReportDisplay({ issues = [] }) {
  const [expandedReports, setExpandedReports] = useState({});
  const [searchTerm, setSearchTerm] = useState('');
  const [filterStatus, setFilterStatus] = useState('all');
  const [sortBy, setSortBy] = useState('newest');
  const [viewMode, setViewMode] = useState('grid');
  const [hoveredCard, setHoveredCard] = useState(null);
  const [selectedReports, setSelectedReports] = useState(new Set());
  
  const toggleReport = (id) => {
    setExpandedReports(prev => ({
      ...prev,
      [id]: !prev[id]
    }));
  };

  const formatDate = (timestamp) => {
    if (!timestamp) return 'No date available';
    const date = new Date(timestamp);
    return date.toLocaleString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const getStatusIcon = (status) => {
    switch (status?.toLowerCase()) {
      case 'resolved':
        return <CheckCircle className="w-4 h-4 text-emerald-400" />;
      case 'in progress':
        return <Clock className="w-4 h-4 text-amber-400" />;
      case 'rejected':
        return <XCircle className="w-4 h-4 text-rose-400" />;
      default:
        return <AlertCircle className="w-4 h-4 text-violet-400" />;
    }
  };

  const getStatusColor = (status) => {
    switch (status?.toLowerCase()) {
      case 'resolved':
        return 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20 shadow-emerald-500/5';
      case 'in progress':
        return 'bg-amber-500/10 text-amber-400 border-amber-500/20 shadow-amber-500/5';
      case 'rejected':
        return 'bg-rose-500/10 text-rose-400 border-rose-500/20 shadow-rose-500/5';
      default:
        return 'bg-violet-500/10 text-violet-400 border-violet-500/20 shadow-violet-500/5';
    }
  };

  const filteredIssues = issues.filter(issue => {
    const matchesSearch = issue.report?.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         issue.category?.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesFilter = filterStatus === 'all' || issue.status?.toLowerCase() === filterStatus;
    return matchesSearch && matchesFilter;
  });

  const sortedIssues = [...filteredIssues].sort((a, b) => {
    switch (sortBy) {
      case 'newest':
        return new Date(b.timestamp) - new Date(a.timestamp);
      case 'oldest':
        return new Date(a.timestamp) - new Date(b.timestamp);
      case 'status':
        return (a.status || '').localeCompare(b.status || '');
      default:
        return 0;
    }
  });

  const statusCounts = issues.reduce((acc, issue) => {
    const status = issue.status?.toLowerCase() || 'pending';
    acc[status] = (acc[status] || 0) + 1;
    return acc;
  }, {});

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 p-6">
      {/* Header Section */}
      <div className="max-w-7xl mx-auto mb-8">
        <div className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl p-6 shadow-2xl">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center gap-4">
              <div className="relative">
                <div className="absolute inset-0 bg-gradient-to-r from-violet-600 to-indigo-600 rounded-xl blur opacity-75"></div>
                <div className="relative bg-gradient-to-r from-violet-600 to-indigo-600 p-3 rounded-xl">
                  <Inbox className="w-8 h-8 text-white" />
                </div>
              </div>
              <div>
                <h1 className="text-3xl font-bold text-white mb-2">Issue Reports Dashboard</h1>
                <p className="text-slate-400">Manage and track all reported issues</p>
              </div>
            </div>
            
            <div className="flex items-center gap-3">
              <button className="bg-white/10 hover:bg-white/20 border border-white/20 text-white px-4 py-2 rounded-lg flex items-center gap-2 transition-all duration-300 hover:scale-105">
                <Download className="w-4 h-4" />
                Export
              </button>
              <button className="bg-gradient-to-r from-violet-600 to-indigo-600 hover:from-violet-700 hover:to-indigo-700 text-white px-6 py-2 rounded-lg font-medium transition-all duration-300 hover:scale-105 shadow-lg">
                New Report
              </button>
            </div>
          </div>

          {/* Stats Cards */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
            {[
              { label: 'Total Reports', value: issues.length, icon: Activity, color: 'from-blue-500 to-cyan-500' },
              { label: 'Resolved', value: statusCounts.resolved || 0, icon: CheckCircle, color: 'from-emerald-500 to-teal-500' },
              { label: 'In Progress', value: statusCounts['in progress'] || 0, icon: Clock, color: 'from-amber-500 to-orange-500' },
              { label: 'Pending', value: statusCounts.pending || 0, icon: AlertCircle, color: 'from-violet-500 to-purple-500' }
            ].map((stat, index) => (
              <div key={index} className="bg-white/5 border border-white/10 rounded-xl p-4 hover:bg-white/10 transition-all duration-300">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-slate-400 text-sm font-medium">{stat.label}</p>
                    <p className="text-2xl font-bold text-white mt-1">{stat.value}</p>
                  </div>
                  <div className={`bg-gradient-to-r ${stat.color} p-2 rounded-lg`}>
                    <stat.icon className="w-5 h-5 text-white" />
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Controls */}
          <div className="flex flex-wrap items-center gap-4">
            <div className="relative flex-1 min-w-64">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-slate-400" />
              <input
                type="text"
                placeholder="Search reports..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full bg-white/10 border border-white/20 text-white placeholder-slate-400 pl-10 pr-4 py-2 rounded-lg focus:outline-none focus:ring-2 focus:ring-violet-500 focus:border-transparent transition-all duration-300"
              />
            </div>
            
            <select
              value={filterStatus}
              onChange={(e) => setFilterStatus(e.target.value)}
              className="bg-white/10 border border-white/20 text-white px-4 py-2 rounded-lg focus:outline-none focus:ring-2 focus:ring-violet-500 appearance-none cursor-pointer"
            >
              <option value="all" className="bg-slate-800">All Status</option>
              <option value="pending" className="bg-slate-800">Pending</option>
              <option value="in progress" className="bg-slate-800">In Progress</option>
              <option value="resolved" className="bg-slate-800">Resolved</option>
              <option value="rejected" className="bg-slate-800">Rejected</option>
            </select>
            
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value)}
              className="bg-white/10 border border-white/20 text-white px-4 py-2 rounded-lg focus:outline-none focus:ring-2 focus:ring-violet-500 appearance-none cursor-pointer"
            >
              <option value="newest" className="bg-slate-800">Newest First</option>
              <option value="oldest" className="bg-slate-800">Oldest First</option>
              <option value="status" className="bg-slate-800">By Status</option>
            </select>
          </div>
        </div>
      </div>

      {/* Reports Section */}
      <div className="max-w-7xl mx-auto">
        {sortedIssues.length === 0 ? (
          <div className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl p-12 text-center shadow-2xl">
            <div className="mb-6">
              <div className="w-24 h-24 bg-gradient-to-r from-violet-500 to-indigo-500 rounded-full flex items-center justify-center mx-auto mb-4 opacity-50">
                <Inbox className="w-12 h-12 text-white" />
              </div>
            </div>
            <h3 className="text-2xl font-bold text-white mb-2">No Reports Found</h3>
            <p className="text-slate-400 mb-6">
              {searchTerm || filterStatus !== 'all' 
                ? "Try adjusting your search or filter criteria" 
                : "Submit your first issue report to see it appear here"}
            </p>
            <button className="bg-gradient-to-r from-violet-600 to-indigo-600 hover:from-violet-700 hover:to-indigo-700 text-white px-6 py-3 rounded-lg font-medium transition-all duration-300 hover:scale-105">
              Create New Report
            </button>
          </div>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
            {sortedIssues.map((issue, index) => (
              <div
                key={issue.id}
                className={`group bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl overflow-hidden shadow-2xl transition-all duration-500 hover:bg-white/10 hover:border-white/20 hover:shadow-3xl ${
                  expandedReports[issue.id] ? 'ring-2 ring-violet-500/50' : ''
                }`}
                onMouseEnter={() => setHoveredCard(issue.id)}
                onMouseLeave={() => setHoveredCard(null)}
                style={{
                  animationDelay: `${index * 0.1}s`
                }}
              >
                {/* Card Header */}
                <div 
                  className="p-6 cursor-pointer"
                  onClick={() => toggleReport(issue.id)}
                >
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex items-center gap-3">
                      <div className="w-10 h-10 bg-gradient-to-r from-violet-600 to-indigo-600 rounded-lg flex items-center justify-center text-white font-bold text-sm">
                        #{issue.id}
                      </div>
                      <div>
                        <h3 className="text-white font-semibold text-lg">Issue #{issue.id}</h3>
                        <div className="flex items-center gap-2 text-slate-400 text-sm mt-1">
                          <Calendar className="w-4 h-4" />
                          <span>{formatDate(issue.timestamp)}</span>
                        </div>
                      </div>
                    </div>
                    
                    <div className="flex items-center gap-2">
                      <div className={`px-3 py-1 rounded-full text-xs font-semibold border flex items-center gap-1.5 ${getStatusColor(issue.status)}`}>
                        {getStatusIcon(issue.status)}
                        <span>{issue.status || 'Pending'}</span>
                      </div>
                      <div className={`transform transition-transform duration-300 ${expandedReports[issue.id] ? 'rotate-180' : ''}`}>
                        <ChevronDown className="w-5 h-5 text-slate-400" />
                      </div>
                    </div>
                  </div>

                  {/* Preview Content */}
                  <div className="space-y-3">
                    {issue.category && (
                      <div className="flex items-center gap-2">
                        <Tag className="w-4 h-4 text-slate-400" />
                        <span className="text-slate-300 text-sm bg-white/10 px-2 py-1 rounded-full">
                          {issue.category}
                        </span>
                      </div>
                    )}
                    
                    <p className="text-slate-300 text-sm line-clamp-2">
                      {issue.report}
                    </p>
                  </div>
                </div>

                {/* Expanded Content */}
                {expandedReports[issue.id] && (
                  <div className="border-t border-white/10 bg-white/5">
                    <div className="p-6 space-y-6">
                      {/* Full Report */}
                      <div>
                        <h4 className="text-white font-semibold mb-3 flex items-center gap-2">
                          <Eye className="w-4 h-4" />
                          Full Report
                        </h4>
                        <div className="bg-black/20 border border-white/10 rounded-lg p-4">
                          <pre className="text-slate-300 text-sm whitespace-pre-wrap font-mono">
                            {issue.report}
                          </pre>
                        </div>
                      </div>

                      {/* Screenshot */}
                      {issue.screenshot && (
                        <div>
                          <h4 className="text-white font-semibold mb-3">Screenshot</h4>
                          <div className="bg-white/5 border border-white/10 rounded-lg overflow-hidden hover:scale-105 transition-transform duration-300 cursor-pointer">
                            <img 
                              src={issue.screenshot} 
                              alt={`Screenshot for issue ${issue.id}`} 
                              className="w-full h-48 object-cover"
                            />
                          </div>
                        </div>
                      )}

                      {/* Actions */}
                      <div className="flex items-center justify-between pt-4 border-t border-white/10">
                        <div className="flex items-center gap-2">
                          <button className="text-slate-400 hover:text-violet-400 p-2 rounded-lg hover:bg-white/10 transition-all duration-300">
                            <Star className="w-4 h-4" />
                          </button>
                          <button className="text-slate-400 hover:text-blue-400 p-2 rounded-lg hover:bg-white/10 transition-all duration-300">
                            <Share2 className="w-4 h-4" />
                          </button>
                          <button className="text-slate-400 hover:text-emerald-400 p-2 rounded-lg hover:bg-white/10 transition-all duration-300">
                            <Edit3 className="w-4 h-4" />
                          </button>
                        </div>
                        
                        <div className="flex items-center gap-2">
                          <button className="text-slate-400 hover:text-rose-400 p-2 rounded-lg hover:bg-white/10 transition-all duration-300">
                            <Trash2 className="w-4 h-4" />
                          </button>
                          <button className="text-slate-400 hover:text-slate-200 p-2 rounded-lg hover:bg-white/10 transition-all duration-300">
                            <MoreHorizontal className="w-4 h-4" />
                          </button>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      <style>{`
        .line-clamp-2 {
          display: -webkit-box;
          -webkit-line-clamp: 2;
          -webkit-box-orient: vertical;
          overflow: hidden;
        }

        @keyframes fadeInUp {
          from {
            opacity: 0;
            transform: translateY(30px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }

        .grid > div {
          animation: fadeInUp 0.6s ease-out forwards;
          opacity: 0;
        }

        .shadow-3xl {
          box-shadow: 0 35px 60px -12px rgba(0, 0, 0, 0.5);
        }

        select option {
          background-color: #1e293b;
          color: white;
        }

        .backdrop-blur-xl {
          backdrop-filter: blur(20px);
        }

        /* Custom scrollbar */
        ::-webkit-scrollbar {
          width: 8px;
        }

        ::-webkit-scrollbar-track {
          background: rgba(255, 255, 255, 0.1);
          border-radius: 4px;
        }

        ::-webkit-scrollbar-thumb {
          background: rgba(139, 92, 246, 0.5);
          border-radius: 4px;
        }

        ::-webkit-scrollbar-thumb:hover {
          background: rgba(139, 92, 246, 0.7);
        }
      `}</style>
    </div>
  );
}

export default ReportDisplay;