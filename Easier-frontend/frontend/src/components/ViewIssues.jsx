import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Container, 
  Typography, 
  Paper, 
  CircularProgress,
  TextField,
  InputAdornment,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip,
  Box,
  IconButton,
  Drawer,
  Divider,
  Grid,
  Card,
  CardContent,
  Tooltip,
  Badge,
  Button,
  Menu,
  Pagination,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TableSortLabel
} from '@mui/material';
import { 
  Search, 
  FilterList, 
  Refresh, 
  Download, 
  MoreVert,
  Visibility,
  Edit,
  LocationOn,
  CalendarToday,
  Info,
  CheckCircle,
  Pending,
  Error,
  Close,
  ArrowUpward,
  ArrowDownward,
  GetApp,
  Settings
} from '@mui/icons-material';
import axios from 'axios';
import './ViewIssues.css';

const ViewIssues = () => {
  const [issues, setIssues] = useState([]);
  const [filteredIssues, setFilteredIssues] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState('all');
  const [sortBy, setSortBy] = useState('id');
  const [sortOrder, setSortOrder] = useState('asc');
  const [page, setPage] = useState(1);
  const [rowsPerPage, setRowsPerPage] = useState(10);
  const [selectedIssue, setSelectedIssue] = useState(null);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [anchorEl, setAnchorEl] = useState(null);
  const open = Boolean(anchorEl);

  // Status options for filter
  const statusOptions = [
    { value: 'all', label: 'All Statuses', color: 'default' },
    { value: 'pending', label: 'Pending', color: 'warning' },
    { value: 'in-progress', label: 'In Progress', color: 'info' },
    { value: 'resolved', label: 'Resolved', color: 'success' },
    { value: 'rejected', label: 'Rejected', color: 'error' }
  ];

  // Fetch issues from API
  useEffect(() => {
    const fetchIssues = async () => {
      try {
        const token = localStorage.getItem('token');
        const response = await axios.get('http://localhost:10000/issues', {
          headers: { Authorization: `Bearer ${token}` }
        });
        setIssues(response.data);
        setFilteredIssues(response.data);
        setLoading(false);
      } catch (err) {
        console.error('Error fetching issues:', err);
        setLoading(false);
      }
    };
    fetchIssues();
  }, []);

  // Apply filters and sorting
  useEffect(() => {
    let result = [...issues];
    
    // Apply search filter
    if (searchTerm) {
      const query = searchTerm.toLowerCase();
      result = result.filter(issue => 
        issue.report_id?.toString().toLowerCase().includes(query) ||
        issue.issue_type?.toLowerCase().includes(query) ||
        issue.status?.toLowerCase().includes(query)
      );
    }
    
    // Apply status filter
    if (statusFilter !== 'all') {
      result = result.filter(issue => issue.status === statusFilter);
    }
    
    // Apply sorting
    result.sort((a, b) => {
      if (sortBy === 'id') {
        return sortOrder === 'asc' 
          ? a.report_id - b.report_id 
          : b.report_id - a.report_id;
      } else if (sortBy === 'timestamp') {
        return sortOrder === 'asc' 
          ? new Date(a.timestamp) - new Date(b.timestamp)
          : new Date(b.timestamp) - new Date(a.timestamp);
      } else if (sortBy === 'status') {
        return sortOrder === 'asc'
          ? a.status.localeCompare(b.status)
          : b.status.localeCompare(a.status);
      }
      return 0;
    });
    
    setFilteredIssues(result);
  }, [issues, searchTerm, statusFilter, sortBy, sortOrder]);

  // Handle sort request
  const handleSort = (property) => {
    const isAsc = sortBy === property && sortOrder === 'asc';
    setSortOrder(isAsc ? 'desc' : 'asc');
    setSortBy(property);
  };

  // Handle page change
  const handleChangePage = (event, newPage) => {
    setPage(newPage);
  };

  // Handle rows per page change
  const handleChangeRowsPerPage = (event) => {
    setRowsPerPage(parseInt(event.target.value, 10));
    setPage(1);
  };

  // Open issue details drawer
  const handleViewIssue = (issue) => {
    setSelectedIssue(issue);
    setDrawerOpen(true);
  };

  // Close drawer
  const handleCloseDrawer = () => {
    setDrawerOpen(false);
    setSelectedIssue(null);
  };

  // Handle menu open
  const handleMenuClick = (event) => {
    setAnchorEl(event.currentTarget);
  };

  // Handle menu close
  const handleMenuClose = () => {
    setAnchorEl(null);
  };

  // Get status icon
  const getStatusIcon = (status) => {
    switch (status) {
      case 'resolved':
        return <CheckCircle color="success" />;
      case 'in-progress':
        return <Pending color="info" />;
      case 'rejected':
        return <Error color="error" />;
      default:
        return <Pending color="warning" />;
    }
  };

  // Get status color
  const getStatusColor = (status) => {
    switch (status) {
      case 'resolved':
        return 'success';
      case 'in-progress':
        return 'info';
      case 'rejected':
        return 'error';
      default:
        return 'warning';
    }
  };

  // Calculate pagination
  const emptyRows = page > 0 ? Math.max(0, (1 + page) * rowsPerPage - filteredIssues.length) : 0;
  const paginatedIssues = filteredIssues.slice(
    (page - 1) * rowsPerPage,
    (page - 1) * rowsPerPage + rowsPerPage
  );

  // Skeleton loader for table rows
  const SkeletonRow = () => (
    <TableRow>
      <TableCell><motion.div animate={{ opacity: [0.5, 1, 0.5] }} transition={{ repeat: Infinity, duration: 1.5 }}><CircularProgress size={20} /></motion.div></TableCell>
      <TableCell><motion.div animate={{ opacity: [0.5, 1, 0.5] }} transition={{ repeat: Infinity, duration: 1.5 }}><CircularProgress size={20} /></motion.div></TableCell>
      <TableCell><motion.div animate={{ opacity: [0.5, 1, 0.5] }} transition={{ repeat: Infinity, duration: 1.5 }}><CircularProgress size={20} /></motion.div></TableCell>
      <TableCell><motion.div animate={{ opacity: [0.5, 1, 0.5] }} transition={{ repeat: Infinity, duration: 1.5 }}><CircularProgress size={20} /></motion.div></TableCell>
    </TableRow>
  );

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
      className="view-issues-container"
    >
      <Container maxWidth="lg" className="container">
        {/* Header Section */}
        <motion.div 
          className="header-section"
          initial={{ y: -20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          <Typography variant="h3" className="page-title">
            <Info className="title-icon" />
            Issue Management
          </Typography>
          
          <div className="header-actions">
            <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
              <Button 
                variant="outlined" 
                startIcon={<Refresh />}
                onClick={() => window.location.reload()}
                className="action-button"
              >
                Refresh
              </Button>
            </motion.div>
            
            <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
              <Button 
                variant="outlined" 
                startIcon={<Download />}
                className="action-button"
              >
                Export
              </Button>
            </motion.div>
            
            <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
              <IconButton onClick={handleMenuClick} className="menu-button">
                <MoreVert />
              </IconButton>
            </motion.div>
            
            <Menu
              anchorEl={anchorEl}
              open={open}
              onClose={handleMenuClose}
              className="header-menu"
            >
              <MenuItem onClick={handleMenuClose}>
                <Settings className="menu-icon" />
                Settings
              </MenuItem>
              <MenuItem onClick={handleMenuClose}>
                <GetApp className="menu-icon" />
                Download All
              </MenuItem>
            </Menu>
          </div>
        </motion.div>
        
        {/* Stats Cards */}
        <motion.div 
          className="stats-container"
          initial={{ y: -20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.3 }}
        >
          <Card className="stat-card">
            <CardContent className="stat-content">
              <Typography variant="h4" className="stat-value">{issues.length}</Typography>
              <Typography variant="body2" className="stat-label">Total Issues</Typography>
            </CardContent>
          </Card>
          
          <Card className="stat-card">
            <CardContent className="stat-content">
              <Typography variant="h4" className="stat-value">
                {issues.filter(i => i.status === 'pending').length}
              </Typography>
              <Typography variant="body2" className="stat-label">Pending</Typography>
            </CardContent>
          </Card>
          
          <Card className="stat-card">
            <CardContent className="stat-content">
              <Typography variant="h4" className="stat-value">
                {issues.filter(i => i.status === 'in-progress').length}
              </Typography>
              <Typography variant="body2" className="stat-label">In Progress</Typography>
            </CardContent>
          </Card>
          
          <Card className="stat-card">
            <CardContent className="stat-content">
              <Typography variant="h4" className="stat-value">
                {issues.filter(i => i.status === 'resolved').length}
              </Typography>
              <Typography variant="body2" className="stat-label">Resolved</Typography>
            </CardContent>
          </Card>
        </motion.div>
        
        {/* Filters Section */}
        <motion.div 
          className="filters-section"
          initial={{ y: -20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.4 }}
        >
          <Paper className="filters-paper" elevation={2}>
            <Grid container spacing={2} alignItems="center">
              <Grid item xs={12} md={6}>
                <TextField
                  fullWidth
                  placeholder="Search issues..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  InputProps={{
                    startAdornment: (
                      <InputAdornment position="start">
                        <Search />
                      </InputAdornment>
                    ),
                  }}
                  className="search-field"
                />
              </Grid>
              
              <Grid item xs={12} md={3}>
                <FormControl fullWidth>
                  <InputLabel>Status</InputLabel>
                  <Select
                    value={statusFilter}
                    onChange={(e) => setStatusFilter(e.target.value)}
                    label="Status"
                    startAdornment={
                      <InputAdornment position="start">
                        <FilterList />
                      </InputAdornment>
                    }
                  >
                    {statusOptions.map((option) => (
                      <MenuItem key={option.value} value={option.value}>
                        <Chip 
                          label={option.label} 
                          color={option.color} 
                          size="small" 
                          className="status-chip"
                        />
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>
              
              <Grid item xs={12} md={3}>
                <Typography variant="body2" className="results-count">
                  Showing {filteredIssues.length} of {issues.length} issues
                </Typography>
              </Grid>
            </Grid>
          </Paper>
        </motion.div>
        
        {/* Table Section */}
        <motion.div 
          className="table-section"
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.5 }}
        >
          <Paper className="table-paper" elevation={3}>
            {loading ? (
              <Box className="loading-container">
                <CircularProgress size={60} thickness={4} />
                <Typography variant="h6" className="loading-text">
                  Loading issues...
                </Typography>
              </Box>
            ) : (
              <>
                <TableContainer>
                  <Table className="issues-table">
                    <TableHead>
                      <TableRow>
                        <TableCell>
                          <TableSortLabel
                            active={sortBy === 'id'}
                            direction={sortBy === 'id' ? sortOrder : 'asc'}
                            onClick={() => handleSort('id')}
                          >
                            ID
                            {sortBy === 'id' ? (
                              sortOrder === 'desc' ? <ArrowDownward /> : <ArrowUpward />
                            ) : null}
                          </TableSortLabel>
                        </TableCell>
                        <TableCell>Issue Type</TableCell>
                        <TableCell>
                          <TableSortLabel
                            active={sortBy === 'timestamp'}
                            direction={sortBy === 'timestamp' ? sortOrder : 'asc'}
                            onClick={() => handleSort('timestamp')}
                          >
                            Timestamp
                            {sortBy === 'timestamp' ? (
                              sortOrder === 'desc' ? <ArrowDownward /> : <ArrowUpward />
                            ) : null}
                          </TableSortLabel>
                        </TableCell>
                        <TableCell>
                          <TableSortLabel
                            active={sortBy === 'status'}
                            direction={sortBy === 'status' ? sortOrder : 'asc'}
                            onClick={() => handleSort('status')}
                          >
                            Status
                            {sortBy === 'status' ? (
                              sortOrder === 'desc' ? <ArrowDownward /> : <ArrowUpward />
                            ) : null}
                          </TableSortLabel>
                        </TableCell>
                        <TableCell align="right">Actions</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      <AnimatePresence>
                        {paginatedIssues.length > 0 ? (
                          paginatedIssues.map((issue) => (
                            <motion.tr
                              key={issue.report_id}
                              component={TableRow}
                              initial={{ opacity: 0, y: 20 }}
                              animate={{ opacity: 1, y: 0 }}
                              exit={{ opacity: 0, y: -20 }}
                              transition={{ duration: 0.3 }}
                              whileHover={{ backgroundColor: 'rgba(0, 0, 0, 0.04)' }}
                              className="issue-row"
                            >
                              <TableCell component="th" scope="row">
                                <Typography variant="body2" className="issue-id">
                                  #{issue.report_id}
                                </Typography>
                              </TableCell>
                              <TableCell>
                                <Typography variant="body2" className="issue-type">
                                  {issue.issue_type}
                                </Typography>
                              </TableCell>
                              <TableCell>
                                <Typography variant="body2" className="issue-date">
                                  {new Date(issue.timestamp).toLocaleDateString()}
                                </Typography>
                              </TableCell>
                              <TableCell>
                                <Chip
                                  icon={getStatusIcon(issue.status)}
                                  label={issue.status.replace('-', ' ')}
                                  color={getStatusColor(issue.status)}
                                  size="small"
                                  className="status-chip"
                                />
                              </TableCell>
                              <TableCell align="right">
                                <Tooltip title="View Details">
                                  <IconButton 
                                    size="small" 
                                    onClick={() => handleViewIssue(issue)}
                                    className="action-icon"
                                  >
                                    <Visibility />
                                  </IconButton>
                                </Tooltip>
                                <Tooltip title="Edit Issue">
                                  <IconButton 
                                    size="small" 
                                    className="action-icon"
                                  >
                                    <Edit />
                                  </IconButton>
                                </Tooltip>
                              </TableCell>
                            </motion.tr>
                          ))
                        ) : (
                          <TableRow>
                            <TableCell colSpan={5} align="center" className="no-results">
                              <Typography variant="body1">
                                No issues found matching your filters
                              </Typography>
                            </TableCell>
                          </TableRow>
                        )}
                      </AnimatePresence>
                      
                      {emptyRows > 0 && (
                        <TableRow style={{ height: 53 * emptyRows }}>
                          <TableCell colSpan={5} />
                        </TableRow>
                      )}
                    </TableBody>
                  </Table>
                </TableContainer>
                
                <Divider />
                
                {/* Pagination */}
                <Box className="pagination-container">
                  <Typography variant="body2" className="pagination-info">
                    Showing {(page - 1) * rowsPerPage + 1} to {Math.min(page * rowsPerPage, filteredIssues.length)} of {filteredIssues.length} issues
                  </Typography>
                  <Pagination
                    count={Math.ceil(filteredIssues.length / rowsPerPage)}
                    page={page}
                    onChange={handleChangePage}
                    rowsPerPage={rowsPerPage}
                    onRowsPerPageChange={handleChangeRowsPerPage}
                    className="pagination"
                  />
                </Box>
              </>
            )}
          </Paper>
        </motion.div>
      </Container>
      
      {/* Issue Details Drawer */}
      <Drawer
        anchor="right"
        open={drawerOpen}
        onClose={handleCloseDrawer}
        className="issue-drawer"
      >
        {selectedIssue && (
          <Box className="drawer-content" p={3}>
            <div className="drawer-header">
              <Typography variant="h5" className="drawer-title">
                Issue #{selectedIssue.report_id}
              </Typography>
              <IconButton onClick={handleCloseDrawer} className="close-button">
                <Close />
              </IconButton>
            </div>
            
            <Divider className="drawer-divider" />
            
            <div className="drawer-section">
              <Typography variant="h6" className="section-title">
                Issue Information
              </Typography>
              
              <Grid container spacing={2}>
                <Grid item xs={12}>
                  <div className="info-item">
                    <Typography variant="body2" className="info-label">
                      Issue Type
                    </Typography>
                    <Typography variant="body1" className="info-value">
                      {selectedIssue.issue_type}
                    </Typography>
                  </div>
                </Grid>
                
                <Grid item xs={12}>
                  <div className="info-item">
                    <Typography variant="body2" className="info-label">
                      Status
                    </Typography>
                    <Chip
                      icon={getStatusIcon(selectedIssue.status)}
                      label={selectedIssue.status.replace('-', ' ')}
                      color={getStatusColor(selectedIssue.status)}
                      size="small"
                      className="status-chip"
                    />
                  </div>
                </Grid>
                
                <Grid item xs={12}>
                  <div className="info-item">
                    <Typography variant="body2" className="info-label">
                      Reported Date
                    </Typography>
                    <Typography variant="body1" className="info-value">
                      {new Date(selectedIssue.timestamp).toLocaleString()}
                    </Typography>
                  </div>
                </Grid>
                
                <Grid item xs={12}>
                  <div className="info-item">
                    <Typography variant="body2" className="info-label">
                      Location
                    </Typography>
                    <Typography variant="body1" className="info-value">
                      {selectedIssue.address || 'No address provided'}
                    </Typography>
                  </div>
                </Grid>
              </Grid>
            </div>
            
            <div className="drawer-section">
              <Typography variant="h6" className="section-title">
                Actions
              </Typography>
              
              <div className="action-buttons">
                <Button 
                  variant="contained" 
                  color="primary" 
                  startIcon={<Edit />}
                  fullWidth
                  className="drawer-button"
                >
                  Edit Issue
                </Button>
                
                <Button 
                  variant="outlined" 
                  color="secondary" 
                  startIcon={<LocationOn />}
                  fullWidth
                  className="drawer-button"
                >
                  View on Map
                </Button>
                
                <Button 
                  variant="outlined" 
                  startIcon={<Download />}
                  fullWidth
                  className="drawer-button"
                >
                  Download Report
                </Button>
              </div>
            </div>
          </Box>
        )}
      </Drawer>
    </motion.div>
  );
};

export default ViewIssues;