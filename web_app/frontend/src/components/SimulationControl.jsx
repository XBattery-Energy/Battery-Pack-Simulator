import React, { useState, useEffect } from 'react'
import {
  Box,
  Paper,
  Typography,
  Button,
  Grid,
  Card,
  CardContent,
  Alert,
  CircularProgress,
  Chip,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Divider,
  Switch,
  FormControlLabel,
} from '@mui/material'
import ExpandMoreIcon from '@mui/icons-material/ExpandMore'
import PlayArrowIcon from '@mui/icons-material/PlayArrow'
import StopIcon from '@mui/icons-material/Stop'
import PauseIcon from '@mui/icons-material/Pause'
import RefreshIcon from '@mui/icons-material/Refresh'
import HistoryIcon from '@mui/icons-material/History'
import axios from 'axios'
import io from 'socket.io-client'

function SimulationControl() {
  const [status, setStatus] = useState({
    running: false,
    paused: false,
    pack_soc: 0.0,
    pack_voltage: 0.0,
    frame_count: 0,
    elapsed_time: 0.0,
  })
  
  // Generate default session name with date
  const generateSessionName = () => {
    const now = new Date()
    const dateStr = now.toISOString().slice(0, 10).replace(/-/g, '') // YYYYMMDD
    const timeStr = now.toTimeString().slice(0, 8).replace(/:/g, '') // HHMMSS
    return `Simulation_${dateStr}_${timeStr}`
  }

  // Session configuration state
  const [sessionSettings, setSessionSettings] = useState({
    session_name: generateSessionName(),
    simulation_mode: '', // Empty - user must select
    initial_soc_pct: 50.0,
    current_amp: 50.0,
    duration_sec: 3600.0,
    target_soc_pct: null, // For charge/discharge cycles
    use_target_soc: false, // Whether to use target SOC or duration
    fault_scenario: '', // Optional, separate from cycle mode
  })
  
  const [faultScenarios, setFaultScenarios] = useState([])
  const [sessionHistory, setSessionHistory] = useState([])
  const [loading, setLoading] = useState(false)
  const [message, setMessage] = useState(null)
  const [socket, setSocket] = useState(null)

  useEffect(() => {
    // Connect to WebSocket
    const newSocket = io('http://localhost:5050')
    setSocket(newSocket)

    newSocket.on('simulation_status', (data) => {
      setStatus(data)
    })

    newSocket.on('simulation_stopped', (data) => {
      setStatus((prev) => ({ ...prev, running: false, paused: false }))
      setMessage({ type: 'info', text: data.message || 'Simulation stopped' })
      // Refresh session history when simulation stops
      loadSessionHistory()
    })

    // Load initial status and session settings
    loadStatus()
    loadSessionSettings()
    loadFaultScenarios()
    loadSessionHistory()

    return () => {
      newSocket.close()
    }
  }, [])

  const loadStatus = async () => {
    try {
      const response = await axios.get('/api/simulation/status')
      setStatus(response.data)
    } catch (error) {
      console.error('Failed to load status:', error)
    }
  }

  const loadSessionSettings = () => {
    try {
      const saved = localStorage.getItem('session_settings')
      if (saved) {
        setSessionSettings(JSON.parse(saved))
      }
    } catch (error) {
      console.error('Failed to load session settings:', error)
    }
  }

  const loadFaultScenarios = async () => {
    try {
      const response = await axios.get('/api/fault-scenarios')
      console.log('Fault scenarios response:', response.data)
      const scenarios = response.data.scenarios || []
      console.log(`Loaded ${scenarios.length} fault scenarios`)
      setFaultScenarios(scenarios)
    } catch (error) {
      console.error('Failed to load fault scenarios:', error)
      console.error('Error details:', error.response?.data || error.message)
      setFaultScenarios([]) // Set empty array on error
    }
  }

  const loadSessionHistory = async () => {
    try {
      const response = await axios.get('/api/sessions?limit=20')
      setSessionHistory(response.data.sessions || [])
    } catch (error) {
      console.error('Failed to load session history:', error)
    }
  }

  const handleSessionChange = (field) => (event) => {
    let value =
      event.target.type === 'checkbox'
        ? event.target.checked
        : event.target.value
    
    // Convert numeric fields to numbers
    const numericFields = ['initial_soc_pct', 'current_amp', 'duration_sec', 'target_soc_pct']
    if (numericFields.includes(field) && value !== '' && value !== null) {
      const numValue = parseFloat(value)
      value = isNaN(numValue) ? value : numValue
    }
    
    setSessionSettings({ ...sessionSettings, [field]: value })
  }

  const handleStart = async () => {
    setLoading(true)
    setMessage(null)

    try {
      // Validate simulation mode is selected (must be 'charge' or 'discharge')
      if (!sessionSettings.simulation_mode || 
          sessionSettings.simulation_mode === '' || 
          (sessionSettings.simulation_mode !== 'charge' && sessionSettings.simulation_mode !== 'discharge')) {
        setMessage({ type: 'error', text: 'Please select a Simulation Mode: Charge or Discharge' })
        setLoading(false)
        return
      }

      // Check if duration is 0 (continuous mode) and warn user
      if (sessionSettings.duration_sec === 0 && !sessionSettings.use_target_soc) {
        const confirmed = window.confirm(
          'Duration is set to 0 (continuous mode). The simulation will run indefinitely until you click Stop. Continue?'
        )
        if (!confirmed) {
          setLoading(false)
          return
        }
      }

      // Auto-generate session name if empty
      let sessionName = sessionSettings.session_name
      if (!sessionName || sessionName.trim() === '') {
        sessionName = generateSessionName()
        setSessionSettings(prev => ({ ...prev, session_name: sessionName }))
      }
      
      // Save current session settings
      localStorage.setItem('session_settings', JSON.stringify({ ...sessionSettings, session_name: sessionName }))
      
      // Load master settings
      const masterSettings = JSON.parse(localStorage.getItem('master_settings') || '{}')
      
      // Ensure required master settings have defaults
      const defaultMasterSettings = {
        cell_capacity_ah: 100.0,
        num_cells: 16,
        temperature_c: 32.0,
        protocol: 'mcu',
        bidirectional: false,
        uart_port: '',
        baudrate: 921600,
        frame_rate_hz: 50.0,
        voltage_noise_mv: 2.0,
        temp_noise_c: 0.5,
        current_noise_ma: 50.0,
      }
      
      // Merge defaults with saved master settings, then with session settings
      const mergedMasterSettings = { ...defaultMasterSettings, ...masterSettings }
      
      // Prepare config based on simulation mode
      let config = { ...mergedMasterSettings, ...sessionSettings, session_name: sessionName }
      
      // Ensure required fields are present
      if (!config.cell_capacity_ah) config.cell_capacity_ah = 100.0
      if (!config.initial_soc_pct) config.initial_soc_pct = 50.0
      if (!config.temperature_c) config.temperature_c = 32.0
      if (config.current_amp === undefined || config.current_amp === null) config.current_amp = 50.0
      
      // Handle charge/discharge cycle modes
      // Frontend sends positive current_amp for both modes
      // Backend converts: charge -> negative (increases SOC), discharge -> positive (decreases SOC)
      if (sessionSettings.simulation_mode === 'charge') {
        config.current_amp = Math.abs(sessionSettings.current_amp) // Send positive, backend converts to negative
        config.simulation_mode = 'charge'
        if (sessionSettings.use_target_soc && sessionSettings.target_soc_pct) {
          config.target_soc_pct = sessionSettings.target_soc_pct
          config.duration_sec = 0 // Use target SOC instead
        }
      } else if (sessionSettings.simulation_mode === 'discharge') {
        config.current_amp = Math.abs(sessionSettings.current_amp) // Send positive, backend keeps positive
        config.simulation_mode = 'discharge'
        if (sessionSettings.use_target_soc && sessionSettings.target_soc_pct) {
          config.target_soc_pct = sessionSettings.target_soc_pct
          config.duration_sec = 0 // Use target SOC instead
        }
      }

      console.log('Starting simulation with config:', config)
      console.log('Config keys:', Object.keys(config))
      console.log('Config values:', Object.values(config))
      
      try {
        const response = await axios.post('/api/simulation/start', config, {
          headers: {
            'Content-Type': 'application/json'
          }
        })
        
        console.log('Response status:', response.status)
        console.log('Response data:', response.data)
        
        if (response.data.success) {
          setMessage({ type: 'success', text: response.data.message })
          loadStatus()
          // Refresh session history after starting
          setTimeout(() => loadSessionHistory(), 1000)
        } else {
          setMessage({ type: 'error', text: response.data.error || 'Unknown error' })
        }
      } catch (axiosError) {
        console.error('Axios error details:', {
          message: axiosError.message,
          response: axiosError.response,
          status: axiosError.response?.status,
          statusText: axiosError.response?.statusText,
          data: axiosError.response?.data,
          config: axiosError.config,
          request: axiosError.request
        })
        throw axiosError // Re-throw to be caught by outer catch
      }
    } catch (error) {
      console.error('Start simulation error:', error)
      const errorMessage = error.response?.data?.error || 
                          error.response?.data?.message ||
                          error.message || 
                          'Failed to start simulation'
      console.error('Error message:', errorMessage)
      setMessage({ 
        type: 'error', 
        text: errorMessage
      })
    } finally {
      setLoading(false)
    }
  }

  const handleStop = async () => {
    setLoading(true)
    try {
      const response = await axios.post('/api/simulation/stop')
      if (response.data.success) {
        setMessage({ type: 'info', text: response.data.message })
        loadStatus()
        // Refresh session history after stopping
        setTimeout(() => loadSessionHistory(), 1000)
      }
    } catch (error) {
      setMessage({ type: 'error', text: 'Failed to stop simulation' })
    } finally {
      setLoading(false)
    }
  }

  const handlePause = async () => {
    setLoading(true)
    try {
      const response = await axios.post('/api/simulation/pause')
      if (response.data.success) {
        setMessage({ type: 'info', text: response.data.message })
        loadStatus()
      }
    } catch (error) {
      setMessage({ type: 'error', text: 'Failed to pause simulation' })
    } finally {
      setLoading(false)
    }
  }

  const handleResume = async () => {
    setLoading(true)
    try {
      const response = await axios.post('/api/simulation/resume')
      if (response.data.success) {
        setMessage({ type: 'info', text: response.data.message })
        loadStatus()
      }
    } catch (error) {
      setMessage({ type: 'error', text: 'Failed to resume simulation' })
    } finally {
      setLoading(false)
    }
  }

  return (
    <Box>
      <Typography variant="h4" gutterBottom fontWeight="bold">
        Simulation Control
      </Typography>
      <Typography variant="body2" color="text.secondary" paragraph>
        Start, stop, pause, and monitor simulation execution
      </Typography>

      {message && (
        <Alert severity={message.type} sx={{ mb: 2 }} onClose={() => setMessage(null)}>
          {message.text}
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* Session Configuration */}
        <Grid item xs={12}>
          <Accordion defaultExpanded>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography variant="h6" fontWeight="bold">
                Session Configuration
              </Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Grid container spacing={2}>
                <Grid item xs={12} sm={6} md={4}>
                  <TextField
                    fullWidth
                    label="Session Name"
                    value={sessionSettings.session_name}
                    onChange={handleSessionChange('session_name')}
                    placeholder="Auto-generated if empty"
                    helperText={`Auto-generated: ${generateSessionName()}`}
                    size="small"
                  />
                </Grid>
                <Grid item xs={12} sm={6} md={4}>
                  <TextField
                    fullWidth
                    label="Initial SOC (%)"
                    type="number"
                    value={sessionSettings.initial_soc_pct}
                    onChange={handleSessionChange('initial_soc_pct')}
                    inputProps={{ step: 0.1, min: 0, max: 100 }}
                    size="small"
                  />
                </Grid>
                <Grid item xs={12} sm={6} md={4}>
                  <FormControl fullWidth size="small" required>
                    <InputLabel>Simulation Mode *</InputLabel>
                    <Select
                      value={sessionSettings.simulation_mode}
                      label="Simulation Mode *"
                      onChange={handleSessionChange('simulation_mode')}
                      required
                      error={!sessionSettings.simulation_mode}
                      displayEmpty={false}
                    >
                      <MenuItem value="charge">Charge Cycle</MenuItem>
                      <MenuItem value="discharge">Discharge Cycle</MenuItem>
                    </Select>
                    {!sessionSettings.simulation_mode && (
                      <Typography variant="caption" color="error" sx={{ mt: 0.5, ml: 1.75 }}>
                        Please select Charge or Discharge
                      </Typography>
                    )}
                  </FormControl>
                </Grid>

                {/* Charge/Discharge Cycle Fields */}
                {(sessionSettings.simulation_mode === 'charge' || sessionSettings.simulation_mode === 'discharge') && (
                  <>
                    <Grid item xs={12} sm={6} md={4}>
                      <TextField
                        fullWidth
                        label="Current (A)"
                        type="number"
                        value={sessionSettings.current_amp}
                        onChange={handleSessionChange('current_amp')}
                        inputProps={{ step: 0.1, min: 0.1 }}
                        helperText={`${sessionSettings.simulation_mode === 'charge' ? 'Charge' : 'Discharge'} current (positive value)`}
                        size="small"
                      />
                    </Grid>
                    <Grid item xs={12} sm={6} md={4}>
                      <FormControlLabel
                        control={
                          <Switch
                            checked={sessionSettings.use_target_soc}
                            onChange={(e) => setSessionSettings({ ...sessionSettings, use_target_soc: e.target.checked })}
                            size="small"
                          />
                        }
                        label="Stop at Target SOC"
                      />
                    </Grid>
                    {sessionSettings.use_target_soc ? (
                      <Grid item xs={12} sm={6} md={4}>
                        <TextField
                          fullWidth
                          label="Target SOC (%)"
                          type="number"
                          value={sessionSettings.target_soc_pct || ''}
                          onChange={(e) => setSessionSettings({ ...sessionSettings, target_soc_pct: parseFloat(e.target.value) || null })}
                          inputProps={{ step: 0.1, min: 0, max: 100 }}
                          helperText={`${sessionSettings.simulation_mode === 'charge' ? 'Charge' : 'Discharge'} until this SOC`}
                          size="small"
                        />
                      </Grid>
                    ) : (
                      <Grid item xs={12} sm={6} md={4}>
                        <TextField
                          fullWidth
                          label="Duration (s)"
                          type="number"
                          value={sessionSettings.duration_sec}
                          onChange={handleSessionChange('duration_sec')}
                          inputProps={{ step: 1, min: 0 }}
                          helperText={sessionSettings.duration_sec === 0 ? "0 = continuous (use Stop button to end)" : "0 = continuous"}
                          size="small"
                          error={sessionSettings.duration_sec === 0}
                        />
                      </Grid>
                    )}
                  </>
                )}

                {/* Fault Scenario - Optional and Separate */}
                <Grid item xs={12}>
                  <Divider sx={{ my: 1 }} />
                  <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                    Fault Injection (Optional)
                  </Typography>
                </Grid>
                <Grid item xs={12} sm={6}>
                  <FormControl fullWidth size="small">
                    <InputLabel>Fault Scenario</InputLabel>
                    <Select
                      value={sessionSettings.fault_scenario}
                      label="Fault Scenario"
                      onChange={handleSessionChange('fault_scenario')}
                    >
                      <MenuItem value="">None</MenuItem>
                      {faultScenarios.map((scenario) => (
                        <MenuItem key={scenario.name} value={scenario.path}>
                          {scenario.name}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Grid>
              </Grid>
            </AccordionDetails>
          </Accordion>
        </Grid>

        {/* Control Buttons */}
        <Grid item xs={12}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom fontWeight="bold">
              Control Actions
            </Typography>
            <Box display="flex" gap={2} flexWrap="wrap">
              <Button
                variant="contained"
                color="success"
                size="large"
                startIcon={loading ? <CircularProgress size={20} color="inherit" /> : <PlayArrowIcon />}
                onClick={handleStart}
                disabled={loading || status.running || !sessionSettings.simulation_mode}
              >
                Start Simulation
              </Button>
              <Button
                variant="contained"
                color="error"
                size="large"
                startIcon={<StopIcon />}
                onClick={handleStop}
                disabled={loading || !status.running}
              >
                Stop
              </Button>
              {status.paused ? (
                <Button
                  variant="contained"
                  color="primary"
                  size="large"
                  startIcon={<PlayArrowIcon />}
                  onClick={handleResume}
                  disabled={loading}
                >
                  Resume
                </Button>
              ) : (
                <Button
                  variant="outlined"
                  color="primary"
                  size="large"
                  startIcon={<PauseIcon />}
                  onClick={handlePause}
                  disabled={loading || !status.running}
                >
                  Pause
                </Button>
              )}
              <Button
                variant="outlined"
                size="large"
                startIcon={<RefreshIcon />}
                onClick={() => {
                  loadStatus()
                  loadSessionHistory()
                }}
                disabled={loading}
              >
                Refresh
              </Button>
            </Box>
          </Paper>
        </Grid>

        {/* Status Cards */}
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="text.secondary" gutterBottom>
                Status
              </Typography>
              <Box display="flex" alignItems="center" gap={1} mt={1}>
                <Chip
                  label={status.running ? (status.paused ? 'Paused' : 'Running') : 'Stopped'}
                  color={status.running ? (status.paused ? 'warning' : 'success') : 'default'}
                  size="small"
                />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="text.secondary" gutterBottom>
                Pack SOC
              </Typography>
              <Typography variant="h4" fontWeight="bold">
                {status.pack_soc.toFixed(2)}%
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="text.secondary" gutterBottom>
                Pack Voltage
              </Typography>
              <Typography variant="h4" fontWeight="bold">
                {status.pack_voltage.toFixed(2)}V
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="text.secondary" gutterBottom>
                Elapsed Time
              </Typography>
              <Typography variant="h4" fontWeight="bold">
                {Math.floor(status.elapsed_time)}s
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        {/* BMS State (if bidirectional) */}
        {status.bms_state && (
          <Grid item xs={12}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom fontWeight="bold">
                BMS State
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={6} sm={3}>
                  <Chip
                    label={`Charge MOSFET: ${status.bms_state.mosfet_charge ? 'ON' : 'OFF'}`}
                    color={status.bms_state.mosfet_charge ? 'success' : 'error'}
                    size="small"
                  />
                </Grid>
                <Grid item xs={6} sm={3}>
                  <Chip
                    label={`Discharge MOSFET: ${status.bms_state.mosfet_discharge ? 'ON' : 'OFF'}`}
                    color={status.bms_state.mosfet_discharge ? 'success' : 'error'}
                    size="small"
                  />
                </Grid>
                <Grid item xs={6} sm={3}>
                  <Chip
                    label={`Protection: ${status.bms_state.protection_active ? 'ACTIVE' : 'INACTIVE'}`}
                    color={status.bms_state.protection_active ? 'error' : 'success'}
                    size="small"
                  />
                </Grid>
                <Grid item xs={6} sm={3}>
                  <Typography variant="body2">
                    BMS Current: {(status.bms_state.bms_current_ma / 1000).toFixed(2)}A
                  </Typography>
                </Grid>
              </Grid>
            </Paper>
          </Grid>
        )}

        {/* Session History */}
        <Grid item xs={12}>
          <Accordion>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Box display="flex" alignItems="center" gap={1}>
                <HistoryIcon />
                <Typography variant="h6" fontWeight="bold">
                  Session History
                </Typography>
                <Chip label={sessionHistory.length} size="small" color="primary" />
              </Box>
            </AccordionSummary>
            <AccordionDetails>
              {sessionHistory.length === 0 ? (
                <Typography variant="body2" color="text.secondary" align="center" sx={{ py: 4 }}>
                  No simulation sessions yet. Start a simulation to create a session.
                </Typography>
              ) : (
                <TableContainer>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>ID</TableCell>
                        <TableCell>Name</TableCell>
                        <TableCell>Start Time</TableCell>
                        <TableCell>End Time</TableCell>
                        <TableCell align="right">Frames</TableCell>
                        <TableCell>Status</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {sessionHistory.map((session) => (
                        <TableRow key={session.id} hover>
                          <TableCell>{session.id}</TableCell>
                          <TableCell>
                            {session.session_name || `Session ${session.id}`}
                          </TableCell>
                          <TableCell>
                            {new Date(session.start_time).toLocaleString()}
                          </TableCell>
                          <TableCell>
                            {session.end_time
                              ? new Date(session.end_time).toLocaleString()
                              : '-'}
                          </TableCell>
                          <TableCell align="right">{session.frame_count || 0}</TableCell>
                          <TableCell>
                            <Chip
                              label={session.status || 'unknown'}
                              color={
                                session.status === 'completed'
                                  ? 'success'
                                  : session.status === 'running'
                                  ? 'warning'
                                  : 'default'
                              }
                              size="small"
                            />
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              )}
            </AccordionDetails>
          </Accordion>
        </Grid>
      </Grid>
    </Box>
  )
}

export default SimulationControl
