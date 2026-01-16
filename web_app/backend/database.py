"""
SQLite Database Module for BMS Data Storage and Analysis

Stores BMS response frames received via bidirectional UART communication.
Provides querying and analysis capabilities.

Uses SQLite for fast local storage - no network latency.
"""

import sqlite3
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any
import threading


class BMSDatabase:
    """SQLite database for storing BMS data."""
    
    def __init__(self, db_path: str = None):
        """
        Initialize BMS database with SQLite.
        
        Args:
            db_path: Path to SQLite database file. Defaults to bms_data.db in backend folder.
        """
        if db_path is None:
            db_path = Path(__file__).parent / "bms_data.db"
        self.db_path = str(db_path)
        self._local = threading.local()
        self._init_database()
        print(f"[DB] SQLite database initialized: {self.db_path}")
    
    def _get_connection(self):
        """Get thread-local database connection."""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn
    
    def _init_database(self):
        """Initialize database schema."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Create simulation_sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS simulation_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_name TEXT,
                start_time TEXT NOT NULL,
                end_time TEXT,
                config TEXT,
                frame_count INTEGER DEFAULT 0,
                status TEXT DEFAULT 'running',
                created_at TEXT NOT NULL
            )
        ''')
        
        # Create bms_frames table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS bms_frames (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_ms INTEGER NOT NULL,
                timestamp_iso TEXT NOT NULL,
                mosfet_status INTEGER NOT NULL,
                protection_flags INTEGER NOT NULL,
                bms_current_ma INTEGER NOT NULL,
                bms_voltage_mv INTEGER NOT NULL,
                balancing_status TEXT NOT NULL,
                fault_codes TEXT NOT NULL,
                bms_state_flags INTEGER NOT NULL,
                mosfet_charge INTEGER NOT NULL,
                mosfet_discharge INTEGER NOT NULL,
                protection_active INTEGER NOT NULL,
                sequence INTEGER,
                session_id INTEGER,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES simulation_sessions(id) ON DELETE CASCADE
            )
        ''')
        
        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_frames_timestamp ON bms_frames(timestamp_ms)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_frames_session ON bms_frames(session_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_sessions_start ON simulation_sessions(start_time)')
        
        # Create fault_events table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS fault_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                timestamp_ms INTEGER NOT NULL,
                timestamp_iso TEXT NOT NULL,
                fault_type TEXT NOT NULL,
                fault_description TEXT,
                cell_index INTEGER,
                severity TEXT,
                resolved INTEGER DEFAULT 0,
                FOREIGN KEY (session_id) REFERENCES simulation_sessions(id) ON DELETE CASCADE
            )
        ''')
        
        conn.commit()
    
    def create_session(self, session_name: Optional[str] = None, config: Optional[Dict] = None) -> int:
        """Create a new simulation session."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        now = datetime.utcnow().isoformat()
        config_json = json.dumps(config) if config else None
        
        cursor.execute('''
            INSERT INTO simulation_sessions 
            (session_name, start_time, config, created_at)
            VALUES (?, ?, ?, ?)
        ''', (session_name, now, config_json, now))
        
        session_id = cursor.lastrowid
        conn.commit()
        print(f"[DB] Created session {session_id}: {session_name}")
        return session_id
    
    def end_session(self, session_id: int, frame_count: int = 0):
        """End a simulation session."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        now = datetime.utcnow().isoformat()
        cursor.execute('''
            UPDATE simulation_sessions 
            SET end_time = ?, frame_count = ?, status = 'completed'
            WHERE id = ?
        ''', (now, frame_count, session_id))
        
        conn.commit()
        print(f"[DB] Ended session {session_id} with {frame_count} frames")
    
    def store_bms_frame(self, bms_data: Dict[str, Any], session_id: Optional[int] = None):
        """Store BMS frame data."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            timestamp_ms = bms_data.get('timestamp_ms', 0)
            timestamp_iso = datetime.utcnow().isoformat()
            
            # Convert numpy arrays to JSON strings
            balancing_status = json.dumps(
                bms_data.get('balancing_status', []).tolist() 
                if isinstance(bms_data.get('balancing_status'), np.ndarray)
                else bms_data.get('balancing_status', [])
            )
            
            fault_codes = json.dumps(
                bms_data.get('fault_codes', []).tolist()
                if isinstance(bms_data.get('fault_codes'), np.ndarray)
                else bms_data.get('fault_codes', [])
            )
            
            cursor.execute('''
                INSERT INTO bms_frames (
                    timestamp_ms, timestamp_iso, mosfet_status, protection_flags,
                    bms_current_ma, bms_voltage_mv, balancing_status, fault_codes,
                    bms_state_flags, mosfet_charge, mosfet_discharge, protection_active,
                    sequence, session_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                timestamp_ms,
                timestamp_iso,
                bms_data.get('mosfet_status', 0),
                bms_data.get('protection_flags', 0),
                bms_data.get('bms_current_ma', 0),
                bms_data.get('bms_voltage_mv', 0),
                balancing_status,
                fault_codes,
                bms_data.get('bms_state_flags', 0),
                1 if bms_data.get('mosfet_charge', False) else 0,
                1 if bms_data.get('mosfet_discharge', False) else 0,
                1 if bms_data.get('protection_active', False) else 0,
                bms_data.get('sequence'),
                session_id,
                timestamp_iso
            ))
            
            conn.commit()
        except Exception as e:
            print(f"[DB] Frame storage failed: {e}")
    
    def get_frames(
        self,
        session_id: Optional[int] = None,
        start_time_ms: Optional[int] = None,
        end_time_ms: Optional[int] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """Query BMS frames."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        query = 'SELECT * FROM bms_frames WHERE 1=1'
        params = []
        
        if session_id:
            query += ' AND session_id = ?'
            params.append(session_id)
        
        if start_time_ms:
            query += ' AND timestamp_ms >= ?'
            params.append(start_time_ms)
        
        if end_time_ms:
            query += ' AND timestamp_ms <= ?'
            params.append(end_time_ms)
        
        query += ' ORDER BY timestamp_ms DESC LIMIT ?'
        params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        frames = []
        for row in rows:
            frame = dict(row)
            if frame.get('balancing_status'):
                try:
                    frame['balancing_status'] = json.loads(frame['balancing_status'])
                except:
                    pass
            if frame.get('fault_codes'):
                try:
                    frame['fault_codes'] = json.loads(frame['fault_codes'])
                except:
                    pass
            frames.append(frame)
        
        return frames
    
    def get_sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get list of simulation sessions."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM simulation_sessions 
            ORDER BY start_time DESC 
            LIMIT ?
        ''', (limit,))
        
        rows = cursor.fetchall()
        sessions = []
        for row in rows:
            session = dict(row)
            if session.get('config'):
                try:
                    session['config'] = json.loads(session['config'])
                except:
                    pass
            sessions.append(session)
        
        print(f"[DB] Retrieved {len(sessions)} sessions")
        return sessions
    
    def get_statistics(self, session_id: Optional[int] = None) -> Dict[str, Any]:
        """Get statistics for BMS data."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if session_id:
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_frames,
                    MIN(timestamp_ms) as min_timestamp,
                    MAX(timestamp_ms) as max_timestamp,
                    AVG(bms_current_ma) as avg_current,
                    AVG(bms_voltage_mv) as avg_voltage,
                    SUM(CASE WHEN protection_active = 1 THEN 1 ELSE 0 END) as protection_events,
                    SUM(CASE WHEN mosfet_charge = 0 THEN 1 ELSE 0 END) as charge_mosfet_opens,
                    SUM(CASE WHEN mosfet_discharge = 0 THEN 1 ELSE 0 END) as discharge_mosfet_opens
                FROM bms_frames
                WHERE session_id = ?
            ''', (session_id,))
        else:
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_frames,
                    MIN(timestamp_ms) as min_timestamp,
                    MAX(timestamp_ms) as max_timestamp,
                    AVG(bms_current_ma) as avg_current,
                    AVG(bms_voltage_mv) as avg_voltage,
                    SUM(CASE WHEN protection_active = 1 THEN 1 ELSE 0 END) as protection_events,
                    SUM(CASE WHEN mosfet_charge = 0 THEN 1 ELSE 0 END) as charge_mosfet_opens,
                    SUM(CASE WHEN mosfet_discharge = 0 THEN 1 ELSE 0 END) as discharge_mosfet_opens
                FROM bms_frames
            ''')
        
        row = cursor.fetchone()
        
        if row:
            return {
                'total_frames': row[0] or 0,
                'min_timestamp_ms': row[1],
                'max_timestamp_ms': row[2],
                'avg_current_ma': float(row[3]) if row[3] else 0.0,
                'avg_voltage_mv': float(row[4]) if row[4] else 0.0,
                'protection_events': row[5] or 0,
                'charge_mosfet_opens': row[6] or 0,
                'discharge_mosfet_opens': row[7] or 0
            }
        else:
            return self._empty_stats()
    
    def _empty_stats(self) -> Dict[str, Any]:
        """Return empty statistics dict."""
        return {
            'total_frames': 0,
            'min_timestamp_ms': None,
            'max_timestamp_ms': None,
            'avg_current_ma': 0.0,
            'avg_voltage_mv': 0.0,
            'protection_events': 0,
            'charge_mosfet_opens': 0,
            'discharge_mosfet_opens': 0
        }
    
    def export_to_csv(self, session_id: Optional[int] = None, output_path: str = "bms_export.csv") -> str:
        """Export BMS data to CSV."""
        import csv
        
        frames = self.get_frames(session_id=session_id, limit=100000)
        
        if not frames:
            raise ValueError("No data to export")
        
        with open(output_path, 'w', newline='') as csvfile:
            fieldnames = [
                'timestamp_ms', 'timestamp_iso', 'mosfet_status', 'protection_flags',
                'bms_current_ma', 'bms_voltage_mv', 'mosfet_charge', 'mosfet_discharge',
                'protection_active', 'sequence'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for frame in frames:
                writer.writerow({
                    'timestamp_ms': frame['timestamp_ms'],
                    'timestamp_iso': frame['timestamp_iso'],
                    'mosfet_status': frame['mosfet_status'],
                    'protection_flags': frame['protection_flags'],
                    'bms_current_ma': frame['bms_current_ma'],
                    'bms_voltage_mv': frame['bms_voltage_mv'],
                    'mosfet_charge': frame['mosfet_charge'],
                    'mosfet_discharge': frame['mosfet_discharge'],
                    'protection_active': frame['protection_active'],
                    'sequence': frame.get('sequence', '')
                })
        
        return output_path
    
    def test_connection(self) -> Dict[str, Any]:
        """Test database connection."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            return {'status': 'connected', 'type': 'sqlite', 'path': self.db_path}
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
