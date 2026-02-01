import { useState, useCallback } from 'react';
import { X, ZoomIn, ZoomOut, RotateCcw, Maximize2 } from 'lucide-react';
import {
    LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
    ResponsiveContainer, ReferenceArea, ReferenceLine
} from 'recharts';

interface ZoomableChartModalProps {
    isOpen: boolean;
    onClose: () => void;
    title: string;
    data: any[];
    lines: {
        dataKey: string;
        stroke: string;
        name?: string;
        strokeWidth?: number;
        strokeDasharray?: string;
        yAxisId?: string;
    }[];
    xAxisKey: string;
    yAxisDomain?: [number | 'auto', number | 'auto'];
    referenceLines?: {
        y: number;
        stroke: string;
        strokeDasharray?: string;
        label?: string;
    }[];
    dualYAxis?: boolean;
}

export default function ZoomableChartModal({
    isOpen,
    onClose,
    title,
    data,
    lines,
    xAxisKey,
    yAxisDomain,
    referenceLines = [],
    dualYAxis = false,
}: ZoomableChartModalProps) {
    // Zoom state
    const [refAreaLeft, setRefAreaLeft] = useState<string | null>(null);
    const [refAreaRight, setRefAreaRight] = useState<string | null>(null);
    const [left, setLeft] = useState<'dataMin' | number>('dataMin');
    const [right, setRight] = useState<'dataMax' | number>('dataMax');
    const [isSelecting, setIsSelecting] = useState(false);

    const handleMouseDown = (e: any) => {
        if (e && e.activeLabel) {
            setRefAreaLeft(e.activeLabel);
            setIsSelecting(true);
        }
    };

    const handleMouseMove = (e: any) => {
        if (isSelecting && e && e.activeLabel) {
            setRefAreaRight(e.activeLabel);
        }
    };

    const handleMouseUp = () => {
        if (!isSelecting) return;
        setIsSelecting(false);

        if (refAreaLeft === refAreaRight || refAreaRight === null) {
            setRefAreaLeft(null);
            setRefAreaRight(null);
            return;
        }

        // Ensure left < right
        let leftVal = Number(refAreaLeft);
        let rightVal = Number(refAreaRight);

        if (leftVal > rightVal) {
            [leftVal, rightVal] = [rightVal, leftVal];
        }

        setLeft(leftVal);
        setRight(rightVal);
        setRefAreaLeft(null);
        setRefAreaRight(null);
    };

    const resetZoom = useCallback(() => {
        setLeft('dataMin');
        setRight('dataMax');
        setRefAreaLeft(null);
        setRefAreaRight(null);
    }, []);

    const zoomOut = useCallback(() => {
        // Expand the view by 50%
        if (typeof left === 'number' && typeof right === 'number') {
            const range = right - left;
            const newLeft = Math.max(0, left - range * 0.5);
            const newRight = right + range * 0.5;

            // Check if we've exceeded original bounds
            const dataMin = Math.min(...data.map(d => d[xAxisKey]));
            const dataMax = Math.max(...data.map(d => d[xAxisKey]));

            if (newLeft <= dataMin && newRight >= dataMax) {
                resetZoom();
            } else {
                setLeft(Math.max(dataMin, newLeft));
                setRight(Math.min(dataMax, newRight));
            }
        }
    }, [left, right, data, xAxisKey, resetZoom]);

    if (!isOpen) return null;

    const isZoomed = left !== 'dataMin' || right !== 'dataMax';

    return (
        <div
            className="fixed inset-0 z-[60] flex items-center justify-center bg-black/50 p-4 backdrop-blur-sm"
            onClick={onClose}
        >
            <div
                className="w-full max-w-5xl max-h-[90vh] bg-white rounded-xl shadow-xl overflow-hidden"
                onClick={(e) => e.stopPropagation()}
            >
                {/* Header */}
                <div className="flex items-center justify-between px-6 py-4 border-b border-notion-border">
                    <div className="flex items-center gap-3">
                        <Maximize2 className="h-5 w-5 text-notion-blue" />
                        <h2 className="text-lg font-semibold text-notion-text">{title}</h2>
                    </div>
                    <div className="flex items-center gap-2">
                        {isZoomed && (
                            <>
                                <button
                                    onClick={zoomOut}
                                    className="flex items-center gap-1.5 rounded-lg border border-notion-border bg-notion-light-gray px-3 py-1.5 text-sm text-notion-text hover:bg-notion-hover transition"
                                    title="Zoom Out"
                                >
                                    <ZoomOut size={16} />
                                    Zoom Out
                                </button>
                                <button
                                    onClick={resetZoom}
                                    className="flex items-center gap-1.5 rounded-lg border border-notion-border bg-notion-light-gray px-3 py-1.5 text-sm text-notion-text hover:bg-notion-hover transition"
                                    title="Reset Zoom"
                                >
                                    <RotateCcw size={16} />
                                    Reset
                                </button>
                            </>
                        )}
                        <button
                            onClick={onClose}
                            className="rounded-lg p-2 text-notion-text-secondary hover:bg-notion-hover hover:text-notion-text transition"
                        >
                            <X size={20} />
                        </button>
                    </div>
                </div>

                {/* Chart */}
                <div className="p-6">
                    <div className="mb-4 text-sm text-notion-text-secondary">
                        <ZoomIn size={14} className="inline mr-1" />
                        Click and drag on the chart to zoom in. Use buttons above to zoom out or reset.
                    </div>

                    <div className="h-[500px]">
                        <ResponsiveContainer width="100%" height="100%">
                            <LineChart
                                data={data}
                                onMouseDown={handleMouseDown}
                                onMouseMove={handleMouseMove}
                                onMouseUp={handleMouseUp}
                                onMouseLeave={handleMouseUp}
                            >
                                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                                <XAxis
                                    dataKey={xAxisKey}
                                    stroke="#94a3b8"
                                    fontSize={12}
                                    domain={[left, right]}
                                    type="number"
                                    allowDataOverflow
                                />
                                {dualYAxis ? (
                                    <>
                                        <YAxis yAxisId="left" stroke="#94a3b8" fontSize={12} domain={yAxisDomain} />
                                        <YAxis yAxisId="right" orientation="right" stroke="#94a3b8" fontSize={12} />
                                    </>
                                ) : (
                                    <YAxis stroke="#94a3b8" fontSize={12} domain={yAxisDomain} />
                                )}
                                <Tooltip
                                    contentStyle={{
                                        backgroundColor: '#ffffff',
                                        border: '1px solid #e5e7eb',
                                        borderRadius: '8px',
                                    }}
                                    labelStyle={{ color: '#374151' }}
                                />
                                <Legend wrapperStyle={{ fontSize: '12px' }} />

                                {/* Reference lines */}
                                {referenceLines.map((ref, idx) => (
                                    <ReferenceLine
                                        key={idx}
                                        y={ref.y}
                                        stroke={ref.stroke}
                                        strokeDasharray={ref.strokeDasharray || '5 5'}
                                        yAxisId={dualYAxis ? 'left' : undefined}
                                        label={ref.label ? { value: ref.label, fill: ref.stroke, fontSize: 12 } : undefined}
                                    />
                                ))}

                                {/* Data lines */}
                                {lines.map((line, idx) => (
                                    <Line
                                        key={idx}
                                        type="monotone"
                                        dataKey={line.dataKey}
                                        stroke={line.stroke}
                                        strokeWidth={line.strokeWidth || 2}
                                        strokeDasharray={line.strokeDasharray}
                                        dot={false}
                                        name={line.name || line.dataKey}
                                        yAxisId={line.yAxisId}
                                    />
                                ))}

                                {/* Selection area for zooming */}
                                {refAreaLeft && refAreaRight && (
                                    <ReferenceArea
                                        x1={refAreaLeft}
                                        x2={refAreaRight}
                                        strokeOpacity={0.3}
                                        fill="#3b82f6"
                                        fillOpacity={0.3}
                                    />
                                )}
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                </div>
            </div>
        </div>
    );
}
