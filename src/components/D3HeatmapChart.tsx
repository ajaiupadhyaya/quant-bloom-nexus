import React, { useRef, useEffect } from 'react';
import * as d3 from 'd3';

interface HeatmapData {
  x: string;
  y: string;
  value: number;
  label?: string;
}

interface D3HeatmapChartProps {
  data: HeatmapData[];
  width?: number;
  height?: number;
  title?: string;
  colorScheme?: string[];
  style?: React.CSSProperties;
}

export const D3HeatmapChart: React.FC<D3HeatmapChartProps> = ({
  data,
  width = 800,
  height = 600,
  title,
  colorScheme = ['#000033', '#000066', '#0000ff', '#0066ff', '#00ccff', '#66ffff'],
  style = {},
}) => {
  const ref = useRef<SVGSVGElement | null>(null);

  useEffect(() => {
    if (!data || data.length === 0) return;

    const svg = d3.select(ref.current);
    svg.selectAll('*').remove();

    const margin = { top: 80, right: 100, bottom: 80, left: 120 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    // Get unique x and y values
    const xValues = Array.from(new Set(data.map(d => d.x))).sort();
    const yValues = Array.from(new Set(data.map(d => d.y))).sort();

    // Calculate cell dimensions
    const cellWidth = innerWidth / xValues.length;
    const cellHeight = innerHeight / yValues.length;

    // Color scale
    const colorScale = d3.scaleSequential()
      .domain(d3.extent(data, d => d.value) as [number, number])
      .interpolator(d3.interpolateViridis);

    // Alternative color scheme
    const customColorScale = d3.scaleLinear<string>()
      .domain(d3.range(0, 1, 1.0 / (colorScheme.length - 1)).concat(1))
      .range(colorScheme);

    // Scales
    const xScale = d3.scaleBand()
      .domain(xValues)
      .range([0, innerWidth])
      .padding(0.1);

    const yScale = d3.scaleBand()
      .domain(yValues)
      .range([0, innerHeight])
      .padding(0.1);

    const valueScale = d3.scaleSequential()
      .domain(d3.extent(data, d => d.value) as [number, number])
      .interpolator(t => customColorScale(t));

    // Main group
    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Create glow filter
    const defs = svg.append('defs');
    const glowFilter = defs.append('filter')
      .attr('id', 'cell-glow')
      .attr('x', '-50%')
      .attr('y', '-50%')
      .attr('width', '200%')
      .attr('height', '200%');

    glowFilter.append('feGaussianBlur')
      .attr('stdDeviation', '2')
      .attr('result', 'coloredBlur');

    const feMerge = glowFilter.append('feMerge');
    feMerge.append('feMergeNode').attr('in', 'coloredBlur');
    feMerge.append('feMergeNode').attr('in', 'SourceGraphic');

    // Tooltip
    const tooltip = d3.select(ref.current?.parentElement)
      .selectAll('.heatmap-tooltip')
      .data([null])
      .join('div')
      .attr('class', 'heatmap-tooltip')
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

    // Create heatmap cells
    g.selectAll('.heatmap-cell')
      .data(data)
      .join('rect')
      .attr('class', 'heatmap-cell')
      .attr('x', d => xScale(d.x) || 0)
      .attr('y', d => yScale(d.y) || 0)
      .attr('width', xScale.bandwidth())
      .attr('height', yScale.bandwidth())
      .attr('fill', d => valueScale(d.value))
      .attr('stroke', '#333')
      .attr('stroke-width', 0.5)
      .attr('rx', 2)
      .attr('ry', 2)
      .on('mouseover', function(event, d) {
        d3.select(this)
          .transition()
          .duration(200)
          .attr('filter', 'url(#cell-glow)')
          .attr('stroke', '#ff6b35')
          .attr('stroke-width', 2);

        tooltip
          .style('display', 'block')
          .html(`
            <div style="border-bottom: 1px solid #ff6b35; margin-bottom: 8px; padding-bottom: 8px;">
              <strong style="color: #ff6b35;">Cell Details</strong>
            </div>
            <div style="margin-bottom: 4px;"><strong>X:</strong> ${d.x}</div>
            <div style="margin-bottom: 4px;"><strong>Y:</strong> ${d.y}</div>
            <div style="margin-bottom: 4px;"><strong>Value:</strong> ${d.value.toFixed(3)}</div>
            ${d.label ? `<div><strong>Label:</strong> ${d.label}</div>` : ''}
          `)
          .style('left', (event.pageX + 15) + 'px')
          .style('top', (event.pageY - 15) + 'px');
      })
      .on('mouseout', function() {
        d3.select(this)
          .transition()
          .duration(200)
          .attr('filter', null)
          .attr('stroke', '#333')
          .attr('stroke-width', 0.5);

        tooltip.style('display', 'none');
      });

    // X-axis
    g.append('g')
      .attr('class', 'x-axis')
      .attr('transform', `translate(0,${innerHeight})`)
      .call(d3.axisBottom(xScale))
      .selectAll('text')
      .attr('fill', '#888')
      .attr('font-size', '11px')
      .style('text-anchor', 'end')
      .attr('dx', '-.8em')
      .attr('dy', '.15em')
      .attr('transform', 'rotate(-45)');

    // Y-axis
    g.append('g')
      .attr('class', 'y-axis')
      .call(d3.axisLeft(yScale))
      .selectAll('text')
      .attr('fill', '#888')
      .attr('font-size', '11px');

    // Remove axis domains
    g.selectAll('.domain').remove();

    // Color legend
    const legendWidth = 20;
    const legendHeight = 200;
    const legendScale = d3.scaleLinear()
      .domain(d3.extent(data, d => d.value) as [number, number])
      .range([legendHeight, 0]);

    const legendG = svg.append('g')
      .attr('transform', `translate(${width - margin.right + 20}, ${margin.top})`);

    // Create gradient for legend
    const legendGradient = defs.append('linearGradient')
      .attr('id', 'legend-gradient')
      .attr('gradientUnits', 'userSpaceOnUse')
      .attr('x1', 0).attr('y1', legendHeight)
      .attr('x2', 0).attr('y2', 0);

    const gradientStops = d3.range(0, 1.1, 0.1);
    gradientStops.forEach(t => {
      legendGradient.append('stop')
        .attr('offset', `${t * 100}%`)
        .attr('stop-color', valueScale(d3.min(data, d => d.value)! + t * (d3.max(data, d => d.value)! - d3.min(data, d => d.value)!)));
    });

    // Legend rectangle
    legendG.append('rect')
      .attr('width', legendWidth)
      .attr('height', legendHeight)
      .attr('fill', 'url(#legend-gradient)')
      .attr('stroke', '#333')
      .attr('stroke-width', 1);

    // Legend axis
    const legendAxis = d3.axisRight(legendScale)
      .tickFormat(d3.format('.2f'))
      .ticks(5);

    legendG.append('g')
      .attr('transform', `translate(${legendWidth}, 0)`)
      .call(legendAxis)
      .selectAll('text')
      .attr('fill', '#888')
      .attr('font-size', '10px');

    legendG.select('.domain').remove();

    // Legend title
    legendG.append('text')
      .attr('x', legendWidth / 2)
      .attr('y', -10)
      .attr('text-anchor', 'middle')
      .attr('font-size', '12px')
      .attr('fill', '#888')
      .text('Value');

    // Chart title
    if (title) {
      svg.append('text')
        .attr('x', width / 2)
        .attr('y', 30)
        .attr('text-anchor', 'middle')
        .attr('font-size', '18px')
        .attr('font-weight', 'bold')
        .attr('fill', '#ff6b35')
        .text(title);
    }

  }, [data, width, height, title, colorScheme]);

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