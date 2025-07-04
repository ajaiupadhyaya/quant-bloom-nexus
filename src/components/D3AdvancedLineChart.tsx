import React, { useRef, useEffect } from 'react';
import * as d3 from 'd3';

interface D3AdvancedLineChartProps {
  data: { x: string | number; y: number; volume?: number }[];
  width?: number;
  height?: number;
  colors?: string[];
  title?: string;
  xLabel?: string;
  yLabel?: string;
  showVolume?: boolean;
  style?: React.CSSProperties;
}

export const D3AdvancedLineChart: React.FC<D3AdvancedLineChartProps> = ({
  data,
  width = 800,
  height = 400,
  colors = ['#ff6b35', '#00d4ff', '#00ff88'],
  title,
  xLabel,
  yLabel,
  showVolume = false,
  style = {},
}) => {
  const ref = useRef<SVGSVGElement | null>(null);

  useEffect(() => {
    if (!data || data.length === 0) return;
    
    const svg = d3.select(ref.current);
    svg.selectAll('*').remove();

    // Enhanced margins for professional layout
    const margin = { top: 60, right: 80, bottom: 80, left: 80 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;
    const volumeHeight = showVolume ? innerHeight * 0.2 : 0;
    const priceHeight = innerHeight - volumeHeight;

    // Create gradients for advanced styling
    const defs = svg.append('defs');
    
    const gradient = defs.append('linearGradient')
      .attr('id', 'area-gradient')
      .attr('gradientUnits', 'userSpaceOnUse')
      .attr('x1', 0).attr('y1', 0)
      .attr('x2', 0).attr('y2', priceHeight);

    gradient.append('stop')
      .attr('offset', '0%')
      .attr('stop-color', colors[0])
      .attr('stop-opacity', 0.8);

    gradient.append('stop')
      .attr('offset', '100%')
      .attr('stop-color', colors[0])
      .attr('stop-opacity', 0.1);

    // Glow filter for advanced effects
    const glowFilter = defs.append('filter')
      .attr('id', 'glow')
      .attr('x', '-20%')
      .attr('y', '-20%')
      .attr('width', '140%')
      .attr('height', '140%');

    glowFilter.append('feGaussianBlur')
      .attr('stdDeviation', '3')
      .attr('result', 'coloredBlur');

    const feMerge = glowFilter.append('feMerge');
    feMerge.append('feMergeNode').attr('in', 'coloredBlur');
    feMerge.append('feMergeNode').attr('in', 'SourceGraphic');

    // Convert data
    const processedData = data.map(d => ({ 
      ...d, 
      x: typeof d.x === 'string' ? d.x : String(d.x),
      parsedX: typeof d.x === 'string' ? new Date(d.x) : new Date(d.x)
    }));

    // Scales
    const xScale = d3.scaleTime()
      .domain(d3.extent(processedData, d => d.parsedX) as [Date, Date])
      .range([0, innerWidth]);

    const yScale = d3.scaleLinear()
      .domain(d3.extent(processedData, d => d.y) as [number, number])
      .nice()
      .range([priceHeight, 0]);

    const volumeScale = d3.scaleLinear()
      .domain([0, d3.max(processedData, d => d.volume || 0) || 1])
      .range([innerHeight, priceHeight + 20]);

    // Main group
    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Advanced grid with subtle styling
    const xTicks = xScale.ticks(8);
    const yTicks = yScale.ticks(6);

    // Vertical grid lines
    g.selectAll('.grid-vertical')
      .data(xTicks)
      .join('line')
      .attr('class', 'grid-vertical')
      .attr('x1', d => xScale(d))
      .attr('x2', d => xScale(d))
      .attr('y1', 0)
      .attr('y2', priceHeight)
      .attr('stroke', '#333')
      .attr('stroke-width', 0.5)
      .attr('opacity', 0.3);

    // Horizontal grid lines
    g.selectAll('.grid-horizontal')
      .data(yTicks)
      .join('line')
      .attr('class', 'grid-horizontal')
      .attr('x1', 0)
      .attr('x2', innerWidth)
      .attr('y1', d => yScale(d))
      .attr('y2', d => yScale(d))
      .attr('stroke', '#333')
      .attr('stroke-width', 0.5)
      .attr('opacity', 0.3);

    // Volume bars if enabled
    if (showVolume) {
      g.selectAll('.volume-bar')
        .data(processedData)
        .join('rect')
        .attr('class', 'volume-bar')
        .attr('x', d => xScale(d.parsedX) - 2)
        .attr('y', d => volumeScale(d.volume || 0))
        .attr('width', 4)
        .attr('height', d => innerHeight - volumeScale(d.volume || 0))
        .attr('fill', colors[2])
        .attr('opacity', 0.4);
    }

    // Area under curve with gradient
    const area = d3.area<any>()
      .x(d => xScale(d.parsedX))
      .y0(priceHeight)
      .y1(d => yScale(d.y))
      .curve(d3.curveCardinal);

    g.append('path')
      .datum(processedData)
      .attr('fill', 'url(#area-gradient)')
      .attr('d', area);

    // Main line with glow effect
    const line = d3.line<any>()
      .x(d => xScale(d.parsedX))
      .y(d => yScale(d.y))
      .curve(d3.curveCardinal);

    g.append('path')
      .datum(processedData)
      .attr('fill', 'none')
      .attr('stroke', colors[0])
      .attr('stroke-width', 3)
      .attr('d', line)
      .attr('filter', 'url(#glow)');

    // Interactive tooltip system
    const tooltip = d3.select(ref.current?.parentElement)
      .selectAll('.advanced-tooltip')
      .data([null])
      .join('div')
      .attr('class', 'advanced-tooltip')
      .style('position', 'absolute')
      .style('background', 'rgba(0, 0, 0, 0.9)')
      .style('border', '1px solid #ff6b35')
      .style('border-radius', '6px')
      .style('padding', '12px')
      .style('font-size', '12px')
      .style('color', '#ffffff')
      .style('box-shadow', '0 4px 20px rgba(255, 107, 53, 0.3)')
      .style('backdrop-filter', 'blur(10px)')
      .style('display', 'none')
      .style('pointer-events', 'none');

    // Crosshair system
    const crosshair = g.append('g')
      .attr('class', 'crosshair')
      .style('display', 'none');

    crosshair.append('line')
      .attr('class', 'crosshair-x')
      .attr('stroke', colors[1])
      .attr('stroke-width', 1)
      .attr('opacity', 0.7)
      .attr('stroke-dasharray', '3,3');

    crosshair.append('line')
      .attr('class', 'crosshair-y')
      .attr('stroke', colors[1])
      .attr('stroke-width', 1)
      .attr('opacity', 0.7)
      .attr('stroke-dasharray', '3,3');

    // Interactive data points
    g.selectAll('.data-point')
      .data(processedData)
      .join('circle')
      .attr('class', 'data-point')
      .attr('cx', d => xScale(d.parsedX))
      .attr('cy', d => yScale(d.y))
      .attr('r', 4)
      .attr('fill', colors[0])
      .attr('stroke', '#000')
      .attr('stroke-width', 2)
      .attr('opacity', 0)
      .on('mouseover', function(event, d) {
        d3.select(this)
          .transition()
          .duration(200)
          .attr('r', 8)
          .attr('opacity', 1);

        // Update crosshair
        crosshair.style('display', null);
        crosshair.select('.crosshair-x')
          .attr('x1', xScale(d.parsedX))
          .attr('x2', xScale(d.parsedX))
          .attr('y1', 0)
          .attr('y2', priceHeight);
        
        crosshair.select('.crosshair-y')
          .attr('x1', 0)
          .attr('x2', innerWidth)
          .attr('y1', yScale(d.y))
          .attr('y2', yScale(d.y));

        // Show tooltip
        tooltip
          .style('display', 'block')
          .html(`
            <div style="border-bottom: 1px solid #ff6b35; margin-bottom: 8px; padding-bottom: 8px;">
              <strong style="color: #ff6b35;">${title || 'Data Point'}</strong>
            </div>
            <div style="margin-bottom: 4px;">
              <strong>${xLabel || 'Time'}:</strong> ${d.parsedX.toLocaleDateString()}
            </div>
            <div style="margin-bottom: 4px;">
              <strong>${yLabel || 'Value'}:</strong> ${d.y.toLocaleString()}
            </div>
            ${d.volume ? `<div><strong>Volume:</strong> ${d.volume.toLocaleString()}</div>` : ''}
          `)
          .style('left', (event.pageX + 15) + 'px')
          .style('top', (event.pageY - 15) + 'px');
      })
      .on('mouseout', function() {
        d3.select(this)
          .transition()
          .duration(200)
          .attr('r', 4)
          .attr('opacity', 0);

        crosshair.style('display', 'none');
        tooltip.style('display', 'none');
      });

    // Professional axes
    const xAxis = d3.axisBottom(xScale)
      .tickFormat(d3.timeFormat('%m/%d'))
      .tickSize(-priceHeight);

    const yAxis = d3.axisLeft(yScale)
      .tickFormat(d => d3.format('.2s')(d))
      .tickSize(-innerWidth);

    g.append('g')
      .attr('class', 'x-axis')
      .attr('transform', `translate(0,${priceHeight})`)
      .call(xAxis)
      .call(g => {
        g.select('.domain').remove();
        g.selectAll('.tick line')
          .attr('stroke', '#333')
          .attr('opacity', 0.3);
        g.selectAll('.tick text')
          .attr('fill', '#888')
          .attr('font-size', '11px');
      });

    g.append('g')
      .attr('class', 'y-axis')
      .call(yAxis)
      .call(g => {
        g.select('.domain').remove();
        g.selectAll('.tick line')
          .attr('stroke', '#333')
          .attr('opacity', 0.3);
        g.selectAll('.tick text')
          .attr('fill', '#888')
          .attr('font-size', '11px');
      });

    // Enhanced title
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

    // Axis labels
    if (xLabel) {
      svg.append('text')
        .attr('x', width / 2)
        .attr('y', height - 20)
        .attr('text-anchor', 'middle')
        .attr('font-size', '12px')
        .attr('fill', '#888')
        .text(xLabel);
    }

    if (yLabel) {
      svg.append('text')
        .attr('transform', 'rotate(-90)')
        .attr('x', -height / 2)
        .attr('y', 20)
        .attr('text-anchor', 'middle')
        .attr('font-size', '12px')
        .attr('fill', '#888')
        .text(yLabel);
    }

  }, [data, width, height, colors, title, xLabel, yLabel, showVolume]);

  return (
    <div style={{ position: 'relative', width, height, ...style }}>
      <svg ref={ref} width={width} height={height} style={{ background: 'linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%)' }} />
    </div>
  );
};