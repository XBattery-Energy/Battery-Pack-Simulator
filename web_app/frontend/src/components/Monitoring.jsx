import React, { useState, useEffect, useRef } from 'react'
import {
  Box,
  Paper,
  Typography,
  Grid,
  Card,
  CardContent,
  Accordion,
  AccordionSummary,
  AccordionDetails,
} from '@mui/material'
import ExpandMoreIcon from '@mui/icons-material/ExpandMore'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ScatterChart,
  Scatter,
} from 'recharts'
import io from 'socket.io-client'

const MAX_DATA_POINTS = 1000

function Monitoring() {
  const [data, setData] = useState([])
  const [currentState, setCurrentState] = useState(null)
  const socketRef = useRef(null)

  useEffect(() => {
    // Connect to WebSocket
    socketRef.current = io('http://localhost:5050')

    socketRef.current.on('simulation_data', (frameData) => {
      setCurrentState(frameData)
      setData((prev) => {
        const newData = [...prev, frameData]
        // Keep only last N points
        return newData.slice(-MAX_DATA_POINTS)
      })
    })

    return () => {
      if (socketRef.current) {
        socketRef.current.close()
      }
    }
  }, [])

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  return (
    <Box>
      <Typography variant="h4" gutterBottom fontWeight="bold">
        Real-Time Monitoring
      </Typography>
      <Typography variant="body2" color="text.secondary" paragraph>
        Live visualization of simulation data
      </Typography>

      {currentState && (
        <Grid container spacing={3} sx={{ mb: 3 }}>
          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent>
                <Typography color="text.secondary" gutterBottom>
                  Pack SOC
                </Typography>
                <Typography variant="h4" fontWeight="bold" color="primary">
                  {currentState.soc_percent.toFixed(2)}%
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
                <Typography variant="h4" fontWeight="bold" color="success.main">
                  {(currentState.pack_voltage_mv / 1000).toFixed(2)}V
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent>
                <Typography color="text.secondary" gutterBottom>
                  Pack Current
                </Typography>
                <Typography variant="h4" fontWeight="bold">
                  {(currentState.pack_current_ma / 1000).toFixed(2)}A
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent>
                <Typography color="text.secondary" gutterBottom>
                  Time
                </Typography>
                <Typography variant="h4" fontWeight="bold">
                  {formatTime(currentState.time_s)}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}

      <Grid container spacing={3}>
        {/* SOC vs Time */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom fontWeight="bold">
              State of Charge
            </Typography>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={data}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis
                  dataKey="time_s"
                  label={{ value: 'Time (s)', position: 'insideBottom', offset: -5 }}
                />
                <YAxis
                  label={{ value: 'SOC (%)', angle: -90, position: 'insideLeft' }}
                />
                <Tooltip />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="soc_percent"
                  stroke="#1976d2"
                  strokeWidth={2}
                  dot={false}
                  name="SOC"
                />
              </LineChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>

        {/* Voltage vs Time */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom fontWeight="bold">
              Pack Voltage
            </Typography>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={data}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis
                  dataKey="time_s"
                  label={{ value: 'Time (s)', position: 'insideBottom', offset: -5 }}
                />
                <YAxis
                  label={{ value: 'Voltage (V)', angle: -90, position: 'insideLeft' }}
                />
                <Tooltip />
                <Legend />
                <Line
                  type="monotone"
                  dataKey={(d) => d.pack_voltage_mv / 1000}
                  stroke="#2e7d32"
                  strokeWidth={2}
                  dot={false}
                  name="Voltage (V)"
                />
              </LineChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>

        {/* Current vs Time */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom fontWeight="bold">
              Pack Current
            </Typography>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={data}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis
                  dataKey="time_s"
                  label={{ value: 'Time (s)', position: 'insideBottom', offset: -5 }}
                />
                <YAxis
                  label={{ value: 'Current (A)', angle: -90, position: 'insideLeft' }}
                />
                <Tooltip />
                <Legend />
                <Line
                  type="monotone"
                  dataKey={(d) => d.pack_current_ma / 1000}
                  stroke="#ed6c02"
                  strokeWidth={2}
                  dot={false}
                  name="Current (A)"
                />
              </LineChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>

        {/* Temperature vs Time */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom fontWeight="bold">
              Average Cell Temperature
            </Typography>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={data}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis
                  dataKey="time_s"
                  label={{ value: 'Time (s)', position: 'insideBottom', offset: -5 }}
                />
                <YAxis
                  label={{ value: 'Temperature (°C)', angle: -90, position: 'insideLeft' }}
                />
                <Tooltip />
                <Legend />
                <Line
                  type="monotone"
                  dataKey={(d) =>
                    d.cell_temperatures_c.reduce((a, b) => a + b, 0) / 16
                  }
                  stroke="#d32f2f"
                  strokeWidth={2}
                  dot={false}
                  name="Avg Temp (°C)"
                />
              </LineChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>

        {/* Cell Voltages Grid */}
        <Grid item xs={12}>
          <Accordion>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography variant="h6" fontWeight="bold">
                Cell Voltages (16 Cells)
              </Typography>
            </AccordionSummary>
            <AccordionDetails>
              <ResponsiveContainer width="100%" height={400}>
                <LineChart data={data}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="time_s" />
                  <YAxis label={{ value: 'Voltage (V)', angle: -90 }} />
                  <Tooltip />
                  <Legend />
                  {currentState &&
                    Array.from({ length: 16 }, (_, i) => (
                      <Line
                        key={i}
                        type="monotone"
                        dataKey={(d) => d.cell_voltages_mv[i] / 1000}
                        stroke={`hsl(${(i * 360) / 16}, 70%, 50%)`}
                        strokeWidth={1}
                        dot={false}
                        name={`Cell ${i + 1}`}
                        connectNulls
                      />
                    ))}
                </LineChart>
              </ResponsiveContainer>
            </AccordionDetails>
          </Accordion>
        </Grid>

        {/* Cell Status Grid */}
        {currentState && (
          <Grid item xs={12}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom fontWeight="bold">
                Cell Status Grid
              </Typography>
              <Grid container spacing={2}>
                {Array.from({ length: 16 }, (_, i) => (
                  <Grid item xs={6} sm={4} md={3} key={i}>
                    <Card variant="outlined">
                      <CardContent>
                        <Typography variant="subtitle2" fontWeight="bold">
                          Cell {i + 1}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          Voltage: {(currentState.cell_voltages_mv[i] / 1000).toFixed(3)}V
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          Temp: {currentState.cell_temperatures_c[i].toFixed(1)}°C
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                ))}
              </Grid>
            </Paper>
          </Grid>
        )}
      </Grid>
    </Box>
  )
}

export default Monitoring
