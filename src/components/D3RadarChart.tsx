import React, { useRef, useEffect } from 'react';
import * as d3 from 'd3';

interface RadarData {
  axis: string;
  value: number;
  max?: number;
}

interface D3RadarChartProps {
  data: RadarData[];
  width?: number;
  height?: number;
  colors?: string[];
  title?: string;
  levels?: number;
  style?: React.CSSProperties;
}

export const D3RadarChart: React.FC<D3RadarChartProps> = ({
  data,
  width = 500,
  height = 500,
  colors = ['#ff6b35', '#00d4ff', '#00ff88'],
  title,
  levels = 5,
  style = {},
}) => {
  const ref = useRef<SVGSVGElement | null>(null);

  useEffect(() => {
    if (!data || data.length === 0) return;

    const svg = d3.select(ref.current);
    svg.selectAll('*').remove();

    const margin = { top: 80, right: 80, bottom: 80, left: 80 };
    const radius = Math.min(width - margin.left - margin.right, height - margin.top - margin.bottom) / 2;
    const centerX = width / 2;
    const centerY = height / 2;

    // Normalize data
    const maxValue = d3.max(data, d => d.max || d.value) || 1;
    const angleSlice = (Math.PI * 2) / data.length;

    // Create scales
    const rScale = d3.scaleLinear()
      .domain([0, maxValue])
      .range([0, radius]);

    // Create radial lines and labels
    const g = svg.append('g')
      .attr('transform', `translate(${centerX},${centerY})`);

    // Create glow filter
    const defs = svg.append('defs');
    const glowFilter = defs.append('filter')
      .attr('id', 'radar-glow')
      .attr('x', '-50%')
      .attr('y', '-50%')
      .attr('width', '200%')
      .attr('height', '200%');

    glowFilter.append('feGaussianBlur')
      .attr('stdDeviation', '3')
      .attr('result', 'coloredBlur');

    const feMerge = glowFilter.append('feMerge');
    feMerge.append('feMergeNode').attr('in', 'coloredBlur');
    feMerge.append('feMergeNode').attr('in', 'SourceGraphic');

    // Create gradient
    const gradient = defs.append('radialGradient')
      .attr('id', 'radar-gradient')
      .attr('cx', '50%')
      .attr('cy', '50%')
      .attr('r', '50%');

    gradient.append('stop')
      .attr('offset', '0%')
      .attr('stop-color', colors[0])
      .attr('stop-opacity', 0.6);

    gradient.append('stop')
      .attr('offset', '100%')
      .attr('stop-color', colors[0])
      .attr('stop-opacity', 0.1);

    // Draw circular grid lines
    for (let level = 1; level <= levels; level++) {
      const levelRadius = (radius / levels) * level;
      
      g.append('circle')
        .attr('cx', 0)
        .attr('cy', 0)
        .attr('r', levelRadius)
        .attr('fill', 'none')
        .attr('stroke', '#333')
        .attr('stroke-width', 1)
        .attr('opacity', 0.3);

      // Add level labels
      if (level === levels) {
        g.append('text')
          .attr('x', 10)
          .attr('y', -levelRadius + 5)
          .attr('font-size', '10px')
          .attr('fill', '#888')
          .text(((maxValue / levels) * level).toFixed(1));
      }
    }

    // Draw axis lines and labels
    data.forEach((d, i) => {
      const angle = angleSlice * i - Math.PI / 2;
      const x = Math.cos(angle) * radius;
      const y = Math.sin(angle) * radius;

      // Axis line
      g.append('line')
        .attr('x1', 0)
        .attr('y1', 0)
        .attr('x2', x)
        .attr('y2', y)
        .attr('stroke', '#333')
        .attr('stroke-width', 1)
        .attr('opacity', 0.5);

      // Axis label
      const labelRadius = radius + 30;
      const labelX = Math.cos(angle) * labelRadius;
      const labelY = Math.sin(angle) * labelRadius;

      g.append('text')
        .attr('x', labelX)
        .attr('y', labelY)
        .attr('text-anchor', 'middle')
        .attr('dominant-baseline', 'middle')
        .attr('font-size', '12px')
        .attr('font-weight', 'bold')
        .attr('fill', colors[1])
        .text(d.axis);
    });

    // Create radar area
    const radarLine = d3.line<RadarData>()
      .x((d, i) => {
        const angle = angleSlice * i - Math.PI / 2;
        return Math.cos(angle) * rScale(d.value);
      })
      .y((d, i) => {
        const angle = angleSlice * i - Math.PI / 2;
        return Math.sin(angle) * rScale(d.value);
      })
      .curve(d3.curveLinearClosed);

    // Draw radar area
    g.append('path')
      .datum(data)
      .attr('d', radarLine)
      .attr('fill', 'url(#radar-gradient)')
      .attr('stroke', colors[0])
      .attr('stroke-width', 3)
      .attr('filter', 'url(#radar-glow)');

    // Tooltip
    const tooltip = d3.select(ref.current?.parentElement)
      .selectAll('.radar-tooltip')
      .data([null])
      .join('div')
      .attr('class', 'radar-tooltip')
      .style('position', 'absolute')
      .style('background', 'rgba(0, 0, 0, 0.9)')
      .style('border', '1px solid #ff6b35')
      .style('border-radius', '6px')
      .style('padding', '12px')
      .style('font-size', '12px')
      .style('color', '#ffffff')
      .style('box-shadow', '0 4px 20px rgba(255, 107, 53, 0.3)')
      .style('backdrop-filter', 'blur(10px)')
      .style('display', 'none');

    // Draw data points
    data.forEach((d, i) => {
      const angle = angleSlice * i - Math.PI / 2;
      const x = Math.cos(angle) * rScale(d.value);
      const y = Math.sin(angle) * rScale(d.value);

      const point = g.append('circle')
        .attr('cx', x)
        .attr('cy', y)
        .attr('r', 6)
        .attr('fill', colors[0])
        .attr('stroke', '#000')
        .attr('stroke-width', 2)
        .style('cursor', 'pointer');

      point.on('mouseover', function(event) {
        d3.select(this)
          .transition()
          .duration(200)
          .attr('r', 10)
          .attr('filter', 'url(#radar-glow)');

        tooltip
          .style('display', 'block')
          .html(`
            <div style="border-bottom: 1px solid #ff6b35; margin-bottom: 8px; padding-bottom: 8px;">
              <strong style="color: #ff6b35;">${d.axis}</strong>
            </div>
            <div style="margin-bottom: 4px;"><strong>Value:</strong> ${d.value.toFixed(2)}</div>
            <div><strong>Max:</strong> ${(d.max || maxValue).toFixed(2)}</div>
            <div style="margin-top: 8px; color: #00d4ff;">
              <strong>Percentage:</strong> ${((d.value / (d.max || maxValue)) * 100).toFixed(1)}%
            </div>
          `)
          .style('left', (event.pageX + 15) + 'px')
          .style('top', (event.pageY - 15) + 'px');
      })
      .on('mouseout', function() {
        d3.select(this)
          .transition()
          .duration(200)
          .attr('r', 6)
          .attr('filter', null);

        tooltip.style('display', 'none');
      });
    });

    // Chart title
    if (title) {
      svg.append('text')
        .attr('x', width / 2)
        .attr('y', 30)
        .attr('text-anchor', 'middle')
        .attr('font-size', '18px')
        .attr('font-weight', 'bold')
        .attr('fill', colors[0])
        .text(title);
    }

    // Legend
    const legend = svg.append('g')
      .attr('transform', `translate(20, ${height - 60})`);

    const legendData = [
      { label: 'Current Values', color: colors[0] },
      { label: 'Axis Labels', color: colors[1] }
    ];

    legendData.forEach((item, i) => {
      const legendItem = legend.append('g')
        .attr('transform', `translate(0, ${i * 20})`);

      legendItem.append('circle')
        .attr('r', 6)
        .attr('fill', item.color);

      legendItem.append('text')
        .attr('x', 15)
        .attr('y', 5)
        .attr('font-size', '12px')
        .attr('fill', '#888')
        .text(item.label);
    });

  }, [data, width, height, colors, title, levels]);

  return (
    <div style={{ position: 'relative', width, height, ...style }}>
      <svg 
        ref={ref} 
        width={width} 
        height={height}
        style={{ 
          background: 'linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%)',
          borderRadius: '4px'
        }}
      />
    </div>
  );
};